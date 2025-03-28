import os
import uuid
import cv2
import numpy as np
import svgwrite
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, Tuple
import logging
import shutil
import math
import networkx as nx
from scipy.spatial.distance import pdist, squareform

# Set up directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
STATIC_DIR = os.path.join(BASE_DIR, "static")
VECTOR_DIR = os.path.join(BASE_DIR, "vectors")
GCODE_DIR = os.path.join(BASE_DIR, "gcode")

# Create directories if they don't exist
for directory in [UPLOAD_DIR, PROCESSED_DIR, STATIC_DIR, VECTOR_DIR, GCODE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tortoisebot-drawing")

# Initialize FastAPI
app = FastAPI(
    title="TortoiseBot2 Drawing System",
    description="Convert images to robot drawing paths",
    version="1.0.0"
)

# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Helper functions for image preprocessing
def preprocess_image(image_path, output_path, target_width=800):
    """
    Preprocess image for the drawing pipeline:
    - Resize to standard width
    - Convert to grayscale
    - Apply bilateral filter
    - Apply adaptive thresholding
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Resize image while preserving aspect ratio
        aspect_ratio = img.shape[0] / img.shape[1]
        target_height = int(target_width * aspect_ratio)
        img = cv2.resize(img, (target_width, target_height))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Save processed image
        cv2.imwrite(output_path, binary)
        
        # Also save edge detection result
        edges = cv2.Canny(blur, 50, 150)
        edge_output_path = os.path.splitext(output_path)[0] + "_edges.png"
        cv2.imwrite(edge_output_path, edges)
        
        return output_path, edge_output_path, img.shape
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

def preprocess_contours(contours, distance_threshold=10):
    """
    Preprocess contours to improve drawing quality:
    - Merge contours that are very close to each other
    - Remove tiny contours that would be hard to draw
    - Sort by size to prioritize larger features
    
    Args:
        contours: List of contours from OpenCV
        distance_threshold: Maximum distance to merge contours
        
    Returns:
        List of preprocessed contours
    """
    if not contours or len(contours) <= 1:
        return contours
    
    # Filter out very small contours
    min_points = 3
    min_area = 30
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if len(contour) >= min_points and area >= min_area:
            filtered_contours.append(contour)
    
    # Find contours that could be merged (endpoints very close to each other)
    merged_contours = []
    used_indices = set()
    
    for i in range(len(filtered_contours)):
        if i in used_indices:
            continue
            
        used_indices.add(i)
        current_contour = filtered_contours[i].copy()
        
        # Try to extend this contour by finding others that connect to it
        while True:
            current_end = current_contour[-1][0]
            found_merge = False
            
            for j in range(len(filtered_contours)):
                if j in used_indices:
                    continue
                
                # Check if the start of contour j is close to the end of current contour
                next_start = filtered_contours[j][0][0]
                if math.dist(current_end, next_start) <= distance_threshold:
                    # Merge contours
                    for point in filtered_contours[j]:
                        current_contour = np.append(current_contour, [point], axis=0)
                    
                    used_indices.add(j)
                    found_merge = True
                    break
                    
                # Check if the end of contour j is close to the end of current contour
                next_end = filtered_contours[j][-1][0]
                if math.dist(current_end, next_end) <= distance_threshold:
                    # Merge contours (reverse the second one)
                    reverse_contour = np.flip(filtered_contours[j], axis=0)
                    for point in reverse_contour:
                        current_contour = np.append(current_contour, [point], axis=0)
                    
                    used_indices.add(j)
                    found_merge = True
                    break
            
            if not found_merge:
                break
        
        merged_contours.append(current_contour)
    
    # Return the merged contours
    logger.info(f"Contour preprocessing: {len(contours)} -> {len(merged_contours)} contours")
    return merged_contours

def detect_contours(binary_image_path, output_path):
    """Extract contours from binary image and visualize them"""
    try:
        # Read binary image
        binary = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter small contours (noise) and apply Douglas-Peucker algorithm
        min_area = 20
        filtered_contours = []
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                # Simplify contours to reduce points (Douglas-Peucker algorithm)
                epsilon = 0.002 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                filtered_contours.append(approx)
        
        # Preprocess contours (merge nearby ones)
        processed_contours = preprocess_contours(filtered_contours, distance_threshold=10)
        
        # Create visualization image
        vis_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_img, processed_contours, -1, (0, 255, 0), 2)
        
        # Save contour visualization
        cv2.imwrite(output_path, vis_img)
        
        logger.info(f"Detected {len(processed_contours)} contours")
        return processed_contours, output_path
    except Exception as e:
        logger.error(f"Contour detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Contour detection failed: {str(e)}")

# Helper functions for vectorization
def contours_to_svg(contours, image_shape, output_path):
    """Convert OpenCV contours to SVG paths"""
    try:
        height, width = image_shape[:2]
        
        # Create SVG drawing
        dwg = svgwrite.Drawing(output_path, size=(width, height))
        
        # Add contours as paths
        for contour in contours:
            if len(contour) < 2:
                continue
                
            # Extract points from contour
            points = []
            for point in contour:
                x, y = point[0]
                points.append((x, y))
                
            # Create SVG path data string
            path_data = f"M {points[0][0]},{points[0][1]}"
            for x, y in points[1:]:
                path_data += f" L {x},{y}"
                
            # Close the path if it's nearly closed
            if len(points) > 2:
                start = np.array(points[0])
                end = np.array(points[-1])
                if np.linalg.norm(end - start) < 10:  # If end point is close to start point
                    path_data += " Z"
                    
            # Add path to SVG
            dwg.add(dwg.path(d=path_data, fill="none", stroke="black", stroke_width=1))
            
        # Save SVG file
        dwg.save()
        
        logger.info(f"Created SVG with {len(contours)} paths: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"SVG creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SVG creation failed: {str(e)}")

def visualize_path_order(contours, path_order, image_shape, output_path):
    """Create a visualization of the optimized path order"""
    try:
        height, width = image_shape[:2]
        
        # Create blank image
        vis_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Colors for visualization - gradient from blue to red
        colors = []
        n_contours = len(contours)
        for i in range(n_contours):
            r = min(255, int(i * 255 / n_contours))
            b = min(255, int(255 - i * 255 / n_contours))
            g = min(r, b) // 2
            colors.append((b, g, r))  # BGR format for OpenCV
        
        # Draw contours in order
        for i, idx in enumerate(path_order):
            contour = contours[idx]
            color = colors[i % len(colors)]
            
            # Draw contour
            cv2.drawContours(vis_img, [contour], 0, color, 2)
            
            # Draw start point
            if len(contour) > 0:
                start_point = tuple(contour[0][0])
                cv2.circle(vis_img, start_point, 5, color, -1)
                
                # Add order number
                cv2.putText(vis_img, str(i+1), start_point, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
        # Save visualization
        cv2.imwrite(output_path, vis_img)
        
        return output_path
    except Exception as e:
        logger.error(f"Path order visualization error: {str(e)}")
        return None  # Non-critical error, return None but don't raise exception

# Path optimization functions
def optimize_path_order(contours) -> Tuple[List[int], List[bool]]:
    """
    Optimize the order of drawing paths to minimize pen travel distance
    
    Returns:
        Tuple containing:
        - List of indices representing the optimal path order
        - List of booleans indicating whether each path should be reversed
    """
    try:
        if not contours or len(contours) <= 1:
            return list(range(len(contours))), [False] * len(contours)
        
        # Calculate start and end points for each contour
        start_points = []
        end_points = []
        
        for contour in contours:
            if len(contour) < 2:
                # For single point contours, use the same point for start and end
                if len(contour) == 1:
                    pt = tuple(contour[0][0])
                    start_points.append(pt)
                    end_points.append(pt)
                continue
                
            # First point is start, last is end
            start_points.append(tuple(contour[0][0]))
            end_points.append(tuple(contour[-1][0]))
        
        # Build a complete graph where:
        # - Nodes are contours
        # - Edge weights are distances between end of one contour and start of another
        G = nx.DiGraph()
        
        for i in range(len(contours)):
            # Add nodes with data about start and end points
            G.add_node(i, start=start_points[i], end=end_points[i])
            
            # Add edges to all other nodes with distances
            for j in range(len(contours)):
                if i != j:
                    # Distance from end of i to start of j
                    forward_dist = math.dist(end_points[i], start_points[j])
                    G.add_edge(i, j, weight=forward_dist)
        
        # Use nearest neighbor algorithm to approximate TSP solution
        path_order = []
        path_reverse = [False] * len(contours)
        
        current_node = 0  # Start with first contour
        path_order.append(current_node)
        unvisited = set(range(1, len(contours)))
        
        while unvisited:
            # Find nearest unvisited contour
            current_end = G.nodes[current_node]['end']
            
            # Calculate distances to all unvisited nodes (considering both directions)
            distances = []
            for next_node in unvisited:
                next_start = G.nodes[next_node]['start']
                next_end = G.nodes[next_node]['end']
                
                # Distance when drawing contour in original direction
                dist_original = math.dist(current_end, next_start)
                
                # Distance when drawing contour in reverse
                dist_reverse = math.dist(current_end, next_end)
                
                distances.append((next_node, False, dist_original))
                distances.append((next_node, True, dist_reverse))
            
            # Find minimum distance
            next_node, reverse, _ = min(distances, key=lambda x: x[2])
            
            # Add to path
            path_order.append(next_node)
            path_reverse[next_node] = reverse
            
            # Update current node and unvisited set
            current_node = next_node
            unvisited.remove(next_node)
        
        logger.info(f"Optimized path: {path_order}")
        return path_order, path_reverse
    
    except Exception as e:
        logger.error(f"Path optimization error: {str(e)}")
        # Return default path if optimization fails
        return list(range(len(contours))), [False] * len(contours)

# G-code generation
def generate_gcode(contours, path_order, path_reverse, image_shape, output_path, 
                  robot_width=100, robot_height=100,  # Reduced default size
                  pen_up=5, pen_down=0,
                  margin_percent=10,  # Margin from edges as percentage
                  travel_speed=3000,
                  drawing_speed=1000,  # Slower drawing speed for better accuracy
                  pen_offset_y=0.033):  # Offset from center of TurtleBot Burger to pen position in meters
    """
    Generate G-code for drawing the optimized contour paths
    
    Args:
        contours: List of contours
        path_order: List of indices indicating the order to draw contours
        path_reverse: List of booleans indicating whether to reverse each contour
        image_shape: Image dimensions (height, width)
        output_path: Path to save the G-code file
        robot_width: Width of robot drawing area in mm
        robot_height: Height of robot drawing area in mm
        pen_up: Z-position for pen up
        pen_down: Z-position for pen down
        margin_percent: Percentage of margin to leave around drawing
        travel_speed: Speed for travel moves (mm/min)
        drawing_speed: Speed for drawing moves (mm/min)
        pen_offset_y: Offset from center of TurtleBot to pen position in meters
    """
    try:
        height, width = image_shape[:2]
        
        # Find bounding box of all contours to center the drawing
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for contour in contours:
            if len(contour) < 1:
                continue
            
            for point in contour:
                x, y = point[0]
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        
        # Calculate drawing dimensions and scale factor
        drawing_width = max_x - min_x
        drawing_height = max_y - min_y
        
        # Calculate margins
        margin_x = robot_width * (margin_percent / 100.0)
        margin_y = robot_height * (margin_percent / 100.0)
        
        # Calculate maximum available space
        available_width = robot_width - 2 * margin_x
        available_height = robot_height - 2 * margin_y
        
        # Calculate scale factor to fit drawing within available space
        scale_x = available_width / drawing_width if drawing_width > 0 else 1.0
        scale_y = available_height / drawing_height if drawing_height > 0 else 1.0
        
        # Use the smaller scale factor to maintain aspect ratio
        scale = min(scale_x, scale_y) * 0.95  # 5% additional safety factor
        
        # Convert pen offset from meters to mm (G-code units)
        pen_offset_mm = pen_offset_y * 1000
        
        logger.info(f"Drawing dimensions: {drawing_width}x{drawing_height} pixels")
        logger.info(f"Available space: {available_width}x{available_height} mm")
        logger.info(f"Scale factor: {scale}")
        logger.info(f"Pen offset: {pen_offset_mm} mm")
        
        # Initialize G-code with header
        gcode = []
        gcode.append("; G-code generated for TortoiseBot2 Drawing System")
        gcode.append("; Original image dimensions: " + str(width) + "x" + str(height) + " pixels")
        gcode.append("; Drawing area: " + str(robot_width) + "x" + str(robot_height) + " mm")
        gcode.append("; Number of paths: " + str(len(path_order)))
        gcode.append("; Scale factor: " + str(scale))
        gcode.append("; Pen offset from center: " + str(pen_offset_mm) + " mm")
        gcode.append("")
        gcode.append("G90 ; Use absolute positioning")
        gcode.append("G21 ; Set units to millimeters")
        gcode.append("G28 ; Home all axes")
        gcode.append(f"G0 Z{pen_up} F1000 ; Raise pen")
        gcode.append(f"G0 F{travel_speed} ; Set travel speed")
        gcode.append("")
        
        # Transform function to convert from image to robot coordinates
        # This now accounts for the pen offset
        def transform_point(x, y):
            # Scale and center the point
            robot_x = margin_x + ((x - min_x) * scale)
            # Invert Y (robot coordinate system has origin at bottom-left)
            robot_y = margin_y + ((max_y - y) * scale)
            
            # Compensate for pen offset - adjust the Y coordinate
            # The robot must position itself so that the pen (offset by pen_offset_mm) 
            # is at the target coordinates
            robot_y -= pen_offset_mm
            
            return robot_x, robot_y
        
        # Generate G-code for each contour in the optimized order
        for i, idx in enumerate(path_order):
            contour = contours[idx]
            reverse = path_reverse[idx]
            
            if len(contour) < 2:
                continue  # Skip very small contours
            
            gcode.append(f"; Path {i+1}/{len(path_order)}")
            
            # Get points in the right order (reversed if needed)
            points = []
            for point in contour:
                x, y = point[0]
                # Transform to robot coordinates
                robot_x, robot_y = transform_point(x, y)
                points.append((robot_x, robot_y))
            
            # Reverse points if needed
            if reverse:
                points = points[::-1]
            
            # Move to start point with pen up
            start_x, start_y = points[0]
            gcode.append(f"G0 X{start_x:.2f} Y{start_y:.2f} ; Move to start of path {i+1}")
            gcode.append(f"G0 Z{pen_down} F1000 ; Lower pen")
            gcode.append(f"G1 F{drawing_speed} ; Set drawing speed")
            
            # Draw the path
            for x, y in points[1:]:
                gcode.append(f"G1 X{x:.2f} Y{y:.2f}")
            
            # Lift pen after path
            gcode.append(f"G0 Z{pen_up} F1000 ; Raise pen")
            gcode.append("")
        
        # Add footer
        gcode.append("; End of drawing")
        gcode.append("G0 Z10 ; Lift pen higher")
        gcode.append("G28 X0 Y0 ; Return to home position")
        
        # Write G-code to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(gcode))
        
        logger.info(f"Generated G-code: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"G-code generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"G-code generation failed: {str(e)}")
# Routes
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the landing page"""
    try:
        html_path = os.path.join(BASE_DIR, "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return HTMLResponse(content="<h1>Error loading page</h1><p>Please check server logs</p>", status_code=500)

@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    robot_width: float = 100,     # Default to 100mm drawing width
    robot_height: float = 100,    # Default to 100mm drawing height
    drawing_speed: int = 1000,    # Default drawing speed
    travel_speed: int = 3000,     # Default travel speed
    pen_offset_y: float = 0.033   # Default 33mm pen offset for TurtleBot Burger
):
    """
    Upload and process an image through the complete drawing pipeline:
    1. Preprocess image
    2. Detect contours
    3. Convert to SVG
    4. Optimize path order
    5. Generate G-code
    
    Returns URLs to all processed outputs
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Only image files allowed. Got: {file.content_type}"
        )
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    upload_filename = f"{file_id}{file_ext}"
    upload_path = os.path.join(UPLOAD_DIR, upload_filename)
    
    # Save uploaded file
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")
    
    # Process the image
    try:
        # Step 1: Preprocess image
        binary_output = os.path.join(PROCESSED_DIR, f"{file_id}_binary.png")
        binary_path, edges_path, img_shape = preprocess_image(upload_path, binary_output)
        
        # Step 2: Detect contours
        contour_output = os.path.join(PROCESSED_DIR, f"{file_id}_contours.png")
        contours, contour_path = detect_contours(binary_path, contour_output)
        
        # Step 3: Convert to SVG
        svg_output = os.path.join(VECTOR_DIR, f"{file_id}.svg")
        svg_path = contours_to_svg(contours, img_shape, svg_output)
        
        # Step 4: Optimize path order
        path_order, path_reverse = optimize_path_order(contours)
        
        # Create path visualization
        path_vis_output = os.path.join(PROCESSED_DIR, f"{file_id}_path_order.png")
        path_vis_path = visualize_path_order(contours, path_order, img_shape, path_vis_output)
        
        # Step 5: Generate G-code with the provided parameters
        gcode_output = os.path.join(GCODE_DIR, f"{file_id}.gcode")
        gcode_path = generate_gcode(
            contours, 
            path_order, 
            path_reverse, 
            img_shape, 
            gcode_output,
            robot_width=robot_width,
            robot_height=robot_height,
            drawing_speed=drawing_speed,
            travel_speed=travel_speed,
            pen_offset_y=pen_offset_y
        )
        
        # Return paths to processed files
        return {
            "originalName": file.filename,
            "processed": {
                "binary": f"/processed/{os.path.basename(binary_path)}",
                "edges": f"/processed/{os.path.basename(edges_path)}",
                "contours": f"/processed/{os.path.basename(contour_path)}",
                "pathOrder": f"/processed/{os.path.basename(path_vis_path)}" if path_vis_path else None,
                "svg": f"/vectors/{os.path.basename(svg_path)}",
                "gcode": f"/gcode/{os.path.basename(gcode_path)}"
            },
            "stats": {
                "numPaths": len(contours),
                "imageSize": {"width": img_shape[1], "height": img_shape[0]},
                "optimalPathLength": len(path_order),
                "drawingSize": {
                    "width": robot_width,
                    "height": robot_height,
                    "units": "mm"
                },
                "penOffset": {
                    "y": pen_offset_y,
                    "units": "m"
                }
            },
            "message": "Image processed successfully"
        }
    except Exception as e:
        logger.error(f"Processing pipeline error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Image processing pipeline failed: {str(e)}"
        )
@app.get("/processed/{filename}")
async def get_processed_image(filename: str):
    """Serve processed images"""
    file_path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@app.get("/vectors/{filename}")
async def get_vector_file(filename: str):
    """Serve vector files (SVG)"""
    file_path = os.path.join(VECTOR_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Vector file not found")
    return FileResponse(file_path)

@app.get("/gcode/{filename}")
async def get_gcode_file(filename: str):
    """Serve G-code files"""
    file_path = os.path.join(GCODE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="G-code file not found")
    return FileResponse(
        file_path,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
