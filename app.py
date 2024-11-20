#set up base path
base_path="."
#Downalod & load RMBG-2.0 Model
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to('cuda')
model.eval()

#Utils
import os
import shutil
import cv2
import numpy as np
from tqdm.autonotebook import tqdm
import re
import uuid
from zipfile import ZipFile
from PIL import Image

def create_directory(directory_path):
  if os.path.exists(directory_path):
    try:
      shutil.rmtree(directory_path)
    except Exception as e:
      print(e)
  os.makedirs(directory_path)

def extract_frames(video_path):
  directory_path = f"{base_path}/images"
  create_directory(directory_path)
  command=f"ffmpeg -i {video_path} {base_path}/images/%07d.png"
  var=os.system(command)

  if var==0:
    print("We extracted frames Successfully")
    print(f"Number of Images {len(os.listdir(directory_path))}")
    return True
  else:
    print("Failed to extract frames")
    print(command)
    return False




def video_size(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    # Get width, height, and FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()

    return (width, height),fps





def make_green_screen(input_image_path):
    ## Generate the save path for the new image
    # base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    # new_file_name = f"{base_name}_remove_background.png"
    # save_image_path = os.path.join(os.path.dirname(input_image_path), new_file_name)

    # Open the input image
    image = Image.open(input_image_path).convert("RGB")  # Ensure image is in RGB mode
    image_size= image.size
    image_size= (1024,1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Transform the image
    input_images = transform_image(image).unsqueeze(0).to('cuda')  # Prepare image for model

    # Predict mask using the model
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    # Convert the prediction to a PIL mask
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)

    # Create a green background image
    green_background = Image.new("RGBA", image.size, (0, 255, 0, 255))  # Green background with full opacity

    # Add the alpha channel (mask) to the original image
    image.putalpha(mask)

    # Composite the green background with the original image using the alpha mask
    final_image = Image.alpha_composite(green_background, image)
    final_image_np = np.array(final_image)
    # If the image has an alpha channel (RGBA), convert it to RGB
    if final_image_np.shape[-1] == 4:  # Check if the image has 4 channels
      final_image_np = final_image_np[:, :, :3]  # Drop the alpha channel
    # Convert RGB to BGR (OpenCV uses BGR format)
    final_image_cv = final_image_np[:, :, ::-1]
    # cv2.imwrite("final_image_cv.png", final_image_cv)
    # Save the resulting image
    # final_image.save(save_image_path)
    # return save_image_path
    return final_image_cv

def process_video(video_path):
  sucess=extract_frames(video_path)
  if sucess==False:
    return
  temp_video_folder = f'{base_path}/video_chunks'
  create_directory(temp_video_folder)
  frames_folder=f"{base_path}/images"
  dir_list = [file for file in os.listdir(frames_folder) if file.endswith(('.jpg', '.png'))]
  dir_list.sort()
  size,fps=video_size(video_path)
  batch = 0
  batchSize = 100
  for i in tqdm(range(0, len(dir_list), batchSize), desc="Processing Batches", unit="batch"):
  # for i in range(0, len(dir_list), batchSize):
    img_array = []
    start, end = i, i + batchSize
    # print("processing ", start, end)
    for filename in dir_list[start:end]:
      filename = frames_folder +"/"+ filename
      img = make_green_screen(filename)
      img_array.append(img)
    # Save the video as MP4
    temp_video_path=temp_video_folder + f'/{str(batch).zfill(4)}.mp4'
    out = cv2.VideoWriter(temp_video_path,
    cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    batch = batch + 1
    for i in range(len(img_array)):
      out.write(img_array[i])
    out.release()
    if os.path.exists(f"{base_path}/gdrive/MyDrive/"):
      drive_folder=f"{base_path}/gdrive/MyDrive/colab/video_chunks"
      os.makedirs(drive_folder, exist_ok=True)
      temp_drive_video_path=f"{drive_folder}" + f'/{str(batch).zfill(4)}.mp4'
      shutil.copy(temp_video_path,temp_drive_video_path)



def make_video(video_path):
  file_name=os.path.basename(video_path).split(".mp4")[0]
  video_folder = f'{base_path}/video_chunks'
  output_txt_file = f'{base_path}/join.txt'
  video_files = [file for file in os.listdir(video_folder) if file.endswith('.mp4')]
  video_files.sort()
  with open(output_txt_file, 'w') as file:
    for video_file in video_files:
      file.write(f"file '{os.path.join(video_folder, video_file)}'\n")
  output_folder=f"{base_path}/result"
  os.makedirs(output_folder, exist_ok=True)
  join_command=f"ffmpeg -f concat -safe 0 -i {base_path}/join.txt -c copy {output_folder}/{file_name}_join.mp4 -y"
  var1=os.system(join_command)
  if var1==0:
    extract_audio_command=f"ffmpeg -i {video_path} {output_folder}/{file_name}.wav -y"
    var2=os.system(extract_audio_command)
    if var2==0:
      add_audio_command=f"ffmpeg -i {output_folder}/{file_name}_join.mp4 -i {output_folder}/{file_name}.wav -c:v copy -map 0:v -map 1:a -y {output_folder}/{file_name}_green_screen.mp4"
      var3=os.system(add_audio_command)
      if var3==0:
        final_result=f"{output_folder}/{file_name}_green_screen.mp4"
        print(f"Green Screen video save at {final_result}")
        return final_result
      else:
        print("Faile to add audio in video")
        print(add_audio_command)
    else:
      print(f"Failed to extract audio")
      print(extract_audio_command)
    return f"{output_folder}/{file_name}_join.mp4"
  else:
    print("Video Marge Failed")
    print(join_command)



def clean_file_name(file_path):
    # Get the base file name and extension
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)

    # Replace non-alphanumeric characters with an underscore
    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)

    # Remove any multiple underscores
    clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')

    # Generate a random UUID for uniqueness
    random_uuid = uuid.uuid4().hex[:6]

    # Combine cleaned file name with the original extension
    clean_file_name=clean_file_name + f"_{random_uuid}" + file_extension
    clean_file_path = os.path.join(os.path.dirname(file_path),clean_file_name )

    return clean_file_path,clean_file_name


def remove_background(upload_image_path):
  upload_folder=f"{base_path}/upload"
  _,clean_name=clean_file_name(upload_image_path)
  input_image_path=f"{upload_folder}/{clean_name}"
  shutil.copy(upload_image_path,input_image_path)
  base_name = os.path.splitext(os.path.basename(input_image_path))[0]  # Get the base name without extension
  new_file_name = f"{base_name}_RMBG.png"  # Append new suffix and change extension
  save_image_path = os.path.join(os.path.dirname(input_image_path), new_file_name)  # Combine with directory
  image = Image.open(input_image_path)
  # image_size = image.size
  image_size = (1024,1024)
  transform_image = transforms.Compose([
      transforms.Resize(image_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  input_images = transform_image(image).unsqueeze(0).to('cuda')
  with torch.no_grad():
      preds = model(input_images)[-1].sigmoid().cpu()
  pred = preds[0].squeeze()
  pred_pil = transforms.ToPILImage()(pred)
  mask = pred_pil.resize(image.size)
  image.putalpha(mask)
  image.save(save_image_path)
  return upload_image_path,save_image_path



def zip_folder(folder_path, zip_path):
    if os.path.exists(zip_path):
      os.remove(zip_path)
    with ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname=arcname)

def manage_files(multiple_images):
  if len(multiple_images)==1:
    _,save_path=remove_background(multiple_images[-1])
    pil_image=Image.open(save_path)
    return save_path,pil_image
  else:
    random_uuid = str(uuid.uuid4().hex)[:6]
    temp_folder=f"{base_path}/temp/RMBG_{random_uuid}"
    os.makedirs(temp_folder)
    for image in multiple_images:
      try:
        upload_image_path,save_path=remove_background(image)
        file_name = os.path.splitext(os.path.basename(upload_image_path))[0]
        shutil.copy(save_path,f"{temp_folder}/{file_name}.png")
      except:
        pass
    zip_folder(temp_folder, f"{temp_folder}.zip")
    full_path = os.path.abspath(f"{temp_folder}.zip")
    return f"{temp_folder}.zip",None

def green_screen_pipeline(upload_video_path):
  upload_folder=f"{base_path}/upload"
  # os.makedirs(upload_folder, exist_ok=True)
  _,clean_name=clean_file_name(upload_video_path)
  video_path=f"{upload_folder}/{clean_name}"
  shutil.copy(upload_video_path,video_path)
  process_video(video_path)
  green_screen_video_path=make_video(video_path)
  return green_screen_video_path
def clear_cache_files():
  folder_list=["images","temp","upload","video_chunks","result"]
  for i in folder_list:
    try:
      shutil.rmtree(f"{base_path}/{i}")
    except:
      pass

#@Gradio Interface
import gradio as gr
import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
  description = """**Note**:<br>
  RMBG v2.0 is available as a source-available model for non-commercial use.<br>
  Developed by: [BRIA AI](https://bria.ai/) HuggingFace model page: [RMBG-2.0 Model](https://huggingface.co/briaai/RMBG-2.0) <br>
  License: [bria-rmbg-2.0](https://bria.ai/bria-huggingface-model-license-agreement/)
  * The model is released under a Creative Commons license for non-commercial use.
  * Commercial use is subject to a commercial agreement with BRIA. Contact BRIA AI for more information.
"""
    # Define Gradio inputs and outputs
  image_demo_inputs=[gr.File(label="Upload Single or Multiple Images",file_count="multiple",file_types=['image'],type='filepath')]
  image_demo_outputs=[gr.File(label="Download Image or Zip File", show_label=True),gr.Image(label="Result")]
  image_demo = gr.Interface(fn=manage_files, inputs=image_demo_inputs,outputs=image_demo_outputs , title="Remove Image Background",description=description)
  video_demo_inputs=[gr.File(label="Upload a Video",file_types=['.mp4'],type='filepath')]
  video_demo_outputs=[gr.File(label="Download Video", show_label=True)]
  video_demo = gr.Interface(fn=green_screen_pipeline, inputs=video_demo_inputs,outputs=video_demo_outputs , title="Remove Video Background (Make Green Screen Video)",description=description)
  demo = gr.TabbedInterface([image_demo,video_demo], ["Remove Image Background", "Remove Video Background"],title="Remove Background on Image & Video Using RMBG-2.0")
  demo.queue().launch(allowed_paths=[f"{base_path}/result",f"{base_path}/upload",f"{base_path}/temp"],debug=debug,share=share)


if __name__ == "__main__":
    # clear_cache_files()
    upload_folder=f"{base_path}/upload"
    os.makedirs(upload_folder, exist_ok=True)
    main()



