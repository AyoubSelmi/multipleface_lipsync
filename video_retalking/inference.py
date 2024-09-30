import numpy as np
import cv2, os, sys, subprocess, platform, torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, 'video_retalking/third_part')
sys.path.insert(0, 'video_retalking/third_part/GPEN')
sys.path.insert(0, 'video_retalking/third_part/GFPGAN')

from video_retalking.third_part.face3d.extract_kp_videos import KeypointExtractor

from video_retalking.utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from video_retalking.utils.inference_utils import face_detect,args
import warnings
warnings.filterwarnings("ignore")


# frames:256x256, full_frames: original size
def datagen(frames, mels, full_frames, frames_pil, all_coordinates,output_folder, base_name):
    img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch = [], [], [], [], [], []
    refs = []
    image_size = 256 
    # original frames
    kp_extractor = KeypointExtractor()
    fr_pil = [Image.fromarray(frame) for frame in frames]
    lms = kp_extractor.extract_keypoint(fr_pil, os.path.join(output_folder,'temp/',base_name+'x12_landmarks.txt'))
    frames_pil = [ (lm, frame) for frame,lm in zip(fr_pil, lms)] # frames is the croped version of modified face
    crops, orig_images , quads  = crop_faces(image_size, frames_pil, scale=1.0, use_fa=True)          
    inverse_transforms = [calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for quad in quads]
    del kp_extractor.detector
    
    face_det_results = face_detect(full_frames, args, jaw_correction=True)
    print("\n\n\n\n")
    print("len face_det_results=", len(face_det_results))
    print("len inverse_transforms=", len(inverse_transforms))
    print("len(crops)=",len(crops))
    print("len(full_frames)=",len(full_frames))
    for idx, (inverse_transform, crop, full_frame, face_det,coordinates) in enumerate(zip(inverse_transforms, crops, full_frames, face_det_results,all_coordinates)):
        oy1= coordinates[1]
        oy2= coordinates[3]
        ox1 = coordinates[0]
        ox2 = coordinates[2]
        print(f"(ox2-ox1,oy2-oy1) = ({ox2-ox1},{oy2-oy1})")
        imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
            cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))                            
        ff = full_frame.copy()        
        print(f"{int(oy1)}:{int(oy2)},{int(ox1)}:{int(ox2)}")
        ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))
        oface, coords = face_det
        y1, y2, x1, x2 = coords
        refs.append(ff[y1: y2, x1:x2])
        imc_pil.save(f"/content/datagen/{idx}_pil_transformed.png")
        cv2.imwrite(f"/content/datagen/{idx}_cropped.png",full_frame[oy1:oy2, ox1:ox2])
        frames_pil[idx][1].save(f"/content/datagen/{idx}_pil.png")
        orig_images[idx].save(f"/content/datagen/{idx}_pil_orig.png")
        crop.save(f"/content/datagen/{idx}_pil_cropped.png")   
        print(f"full frame shape={ff.shape},\
              shape of full frame crop={full_frame[oy1:oy2, ox1:ox2].shape},\
              shape of face det crop={crop.size},\
              shape of transformed crop={imc_pil.size},\
              shape of frame to extract face from={frames_pil[idx][1].size},\
              orig image shape ={orig_images[idx].size}")
    print("len(refs)=",len(refs))
    print("len(frames)=",len(frames))
    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        print("idx = ",idx)
        frame_to_save = frames[idx].copy()        
          
        face = refs[idx]
        oface, coords = face_det_results[idx].copy()
        face = cv2.resize(face, (args.img_size, args.img_size))
        oface = cv2.resize(oface, (args.img_size, args.img_size))

        img_batch.append(oface)
        ref_batch.append(face) 
        mel_batch.append(m)
        coords_batch.append(coords)
        frame_batch.append(frame_to_save)
        full_frame_batch.append(full_frames[idx].copy())        

        if len(img_batch) >= args.LNet_batch_size:
            img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
            img_masked = img_batch.copy()
            img_original = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch
            img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch, ref_batch  = [], [], [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
        img_masked = img_batch.copy()
        img_original = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch