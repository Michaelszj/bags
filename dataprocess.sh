#!/bin/bash

# Set the path to the datasource directory
datasource="datasource"

# Loop through all the .mp4 files in the datasource directory
for video in "$datasource"/*.mp4; do
    # Get the filename without the extension
    filename=$(basename "$video" .mp4)
    
    # Create a new folder with the same name as the video file
    mkdir -p "$datasource/$filename"
    mkdir -p "$datasource/$filename/imgs"
    mkdir -p "$datasource/$filename/masks"
    # mp4e the video file into the new folder
    mv "$video" "$datasource/$filename/"
    
    # Use ffmpeg to extract frames from the video with a frame rate of 10
    ffmpeg -i "$datasource/$filename/$filename.mp4" -vf fps=10 "$datasource/$filename/imgs/%05d.jpg"
    
    # Count the number of frames extracted
    frame_count=$(find "$datasource/$filename/imgs" -type f -name "*.jpg" | wc -l)
    
    # Save the frame count to frames.txt
    echo "$frame_count" > "$datasource/$filename/frames.txt"
done
# for video in "$datasource"/*.MOV; do
#     # Get the filename without the extension
#     filename=$(basename "$video" .MOV)
    
#     # Create a new folder with the same name as the video file
#     mkdir -p "$datasource/$filename"
#     mkdir -p "$datasource/$filename/imgs"
#     mkdir -p "$datasource/$filename/masks"
#     # mp4e the video file into the new folder
#     mv "$video" "$datasource/$filename/"
    
#     # Use ffmpeg to extract frames from the video with a frame rate of 10
#     ffmpeg -i "$datasource/$filename/$filename.MOV" -vf fps=10 "$datasource/$filename/imgs/%05d.jpg"
    
#     # Count the number of frames extracted
#     frame_count=$(find "$datasource/$filename/imgs" -type f -name "*.jpg" | wc -l)
    
#     # Save the frame count to frames.txt
#     echo "$frame_count" > "$datasource/$filename/frames.txt"
# done
# done



#!/bin/bash

# Set the path to the datasource directory
# datasource="datasource"

# # Initialize variables
# total_files=0
# folder_count=0

# # Loop through all the folders named 'datasource/*/imgs'
# for folder in "$datasource"/*/imgs; do
#     # Count the number of files with the postfix '.jpg' in each folder
#     file_count=$(find "$folder" -type f -name "*.jpg" | wc -l)
    
#     # Add the file count to the total files count
#     total_files=$((total_files + file_count))
    
#     # Increment the folder count
#     folder_count=$((folder_count + 1))
# done

# # Calculate the average number of files
# if [ "$folder_count" -gt 0 ]; then
#     average_files=$((total_files / folder_count))
# else
#     average_files=0
# fi

# echo "Average number of files with postfix '.jpg' among folders named as 'datasource/*/imgs': $average_files"

#!/bin/bash

# Set the path to the datasource directory
# datasource="datasource"

# Loop through all the folders under 'datasource'
# for folder in "$datasource"/*/; do
#     # Check if there is no .mp4 file in the folder
#     if ! ls "$folder"/*.mp4 >/dev/null 2>&1; then
#         # Get the folder name
#         folder_name=$(basename "$folder")
        
#         # Use ffmpeg to create a video with a frame rate of 10 from the images in 'imgs' folder
#         ffmpeg -framerate 10 -i "$folder/imgs/%05d.jpg" "$folder/$folder_name.mp4"
#     fi
# done

