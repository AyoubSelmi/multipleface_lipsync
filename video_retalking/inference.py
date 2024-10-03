import numpy as np
import cv2, sys
from PIL import Image

sys.path.insert(0, 'video_retalking/third_part')
sys.path.insert(0, 'video_retalking/third_part/GPEN')
sys.path.insert(0, 'video_retalking/third_part/GFPGAN')

from video_retalking.utils.inference_utils import args
import warnings
warnings.filterwarnings("ignore")


# frames:256x256, full_frames: original size
def datagen(frames, mels, full_frames, frames_pil, all_coordinates,output_folder, base_name):
    img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch = [], [], [], [], [], []
    refs = []
    
    # original frames    
    frames_pil = [Image.fromarray(frame) for frame in frames]
            
    
    for idx, (crop, full_frame,coordinates) in enumerate(zip(frames_pil, full_frames,all_coordinates)):
        oy1= coordinates[1]
        oy2= coordinates[3]
        ox1 = coordinates[0]
        ox2 = coordinates[2]                        
        ff = full_frame.copy()        
        ff[oy1:oy2, ox1:ox2] = cv2.resize(np.array(crop.convert('RGB')), (ox2 - ox1, oy2 - oy1))
        refs.append(ff[oy1:oy2, ox1:ox2])

        
    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)        
        frame_to_save = frames[idx].copy()        
        face = refs [idx]                          

        face = cv2.resize(face, (args.img_size, args.img_size))
        oface = cv2.resize(np.array(full_frames[idx][all_coordinates[idx][1]:all_coordinates[idx][3],all_coordinates[idx][0]:all_coordinates[idx][2]]), (args.img_size, args.img_size))        
        img_batch.append(oface)
        ref_batch.append(face) 
        mel_batch.append(m)
        coords_batch.append([all_coordinates[idx][1],all_coordinates[idx][3],all_coordinates[idx][0],all_coordinates[idx][2]])
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
            img_batch, mel_batch, frame_batch, coords_baFtch, img_original, full_frame_batch, ref_batch  = [], [], [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
        img_masked = img_batch.copy()
        img_original = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch