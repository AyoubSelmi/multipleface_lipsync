import numpy as np
import cv2, os, subprocess, platform, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
from video_retalking.utils import audio
from video_retalking.utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, load_model, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict

# 3dmm extraction
from video_retalking.third_part.face3d.util.preprocess import align_img
from video_retalking.third_part.face3d.util.load_mats import load_lm3d
from video_retalking.third_part.face3d.extract_kp_videos import KeypointExtractor
from video_retalking.inference import datagen
from video_retalking.third_part.ganimation_replicate.model.ganimation import GANimationModel

def find_ordered_sequences_with_status(all_frames, asd_output_frames):
    missing = sorted(set(all_frames) - set(asd_output_frames))      # Find missing values
    non_missing = sorted(set(asd_output_frames))              # Sort the non-missing values
    all_sequences = []
    
    # Helper function to extract sequences
    def extract_sequences(values):
        sequences = []
        current_sequence = []

        for i in range(len(values)):
            # If it's the first element or it's consecutive to the previous element
            if i == 0 or values[i] == values[i-1] + 1:
                current_sequence.append(values[i])
            else:
                # If not consecutive, store the current sequence and start a new one
                sequences.append(current_sequence)
                current_sequence = [values[i]]

        if current_sequence:  # Append the last sequence
            sequences.append(current_sequence)

        return sequences

    # Get missing and non-missing sequences
    missing_sequences = extract_sequences(missing)
    non_missing_sequences = extract_sequences(non_missing)
    
    # Pointers for iterating through missing and non-missing sequences
    i, j = 0, 0

    # Traverse through full and append either missing or non-missing sequences in order
    while i < len(missing_sequences) or j < len(non_missing_sequences):
        if j < len(non_missing_sequences) and (i == len(missing_sequences) or non_missing_sequences[j][0] < missing_sequences[i][0]):
            all_sequences.append((non_missing_sequences[j], True))  # sequence contain a face
            j += 1
        else:
            all_sequences.append((missing_sequences[i], False))  #  if missing this means it does not contain a face
            i += 1

    return all_sequences

def lipsync(enhancer,restorer,fps,full_frames,asd_output,sequence,sequence_idx,output_folder,base_name,audio_path,outfile,lipsync_options,device):
    print(f"lipsyincing sequence {sequence_idx} ")
    print ("[Step 0] Number of frames available for inference: "+str(len(sequence)))        
    # crop face of speaking person
    frames_pil = []
    asd_coordinates = []    
    for fidx in sequence: # frame is a frame containing a face:        
        print("fidx=",fidx)
        bbox = asd_output[fidx]["bbox"]        
        asd_coordinates.append([coordinate for coordinate in bbox])                
        image = full_frames[fidx]        
        print(f"shape of original image = {image.shape}")
        print(f"shape of bbox used for cropping the full frame = {image[bbox[1]:bbox[3],bbox[0]:bbox[2]].shape}")

        cv2.imwrite(f"/content/comparison/{fidx}_original.png",image)                        
        cv2.imwrite(f"/content/comparison/{fidx}_bbox.png",image[bbox[1]:bbox[3],bbox[0]:bbox[2]])                        
        frames_pil.append(Image.fromarray(cv2.resize(image[bbox[1]:bbox[3],bbox[0]:bbox[2]],(256,256))))            
    full_frames = full_frames[sequence[0]:sequence[-1]+1]
    print(f"len(full_frames)={len(full_frames)}")
    print(f"len(frames_pil)={len(frames_pil)}")
    # get the landmark according to the detected face.
    if not os.path.isfile(os.path.join(output_folder,'temp/',base_name+str(sequence_idx)+'_landmarks.txt')):
        print('[Step 1] Landmarks Extraction in Video.')
        kp_extractor = KeypointExtractor()
        print("len of frames to extract landmarks from = ",len(frames_pil))        
        lm = kp_extractor.extract_keypoint(frames_pil, os.path.join(output_folder,'temp/',base_name+str(sequence_idx)+'_landmarks.txt'))
    else:
        print('[Step 1] Using saved landmarks.')
        lm = np.loadtxt(os.path.join(output_folder,'temp/',base_name+str(sequence_idx)+'_landmarks.txt')).astype(np.float32)
        lm = lm.reshape([len(full_frames), -1, 2])
       
    if not os.path.isfile(os.path.join(output_folder,'temp/',base_name+str(sequence)+'_coeffs.npy')):
        net_recon = load_face3d_net(lipsync_options.face3d_net_path, device)
        lm3d_std = load_lm3d('video_retalking/checkpoints/BFM')

        video_coeffs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"):
            frame = frames_pil[idx]
            W, H = frame.size
            lm_idx = lm[idx].reshape([-1, 2])
            if np.mean(lm_idx) == -1:
                lm_idx = (lm3d_std[:, :2]+1) / 2.
                lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
            else:
                lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

            trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, lm3d_std)
            trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_idx_tensor = torch.tensor(np.array(im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0) 
            with torch.no_grad():
                coeffs = split_coeff(net_recon(im_idx_tensor))

            pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'],\
                                         pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
            video_coeffs.append(pred_coeff)
        semantic_npy = np.array(video_coeffs)[:,0]
        np.save(os.path.join(output_folder,'temp/',base_name+str(sequence_idx)+'_coeffs.npy'), semantic_npy)
    else:
        print('[Step 2] Using saved coeffs.')
        semantic_npy = np.load(os.path.join(output_folder,'temp/',base_name+str(sequence)+'_coeffs.npy')).astype(np.float32)

    # generate the 3dmm coeff from a single image    
    print('using expression center')    
    expression = torch.tensor(loadmat('video_retalking/checkpoints/expression.mat')['expression_center'])[0]

    # load DNet, model(LNet and ENet)
    D_Net, model = load_model(lipsync_options, device)

    if not os.path.isfile(os.path.join(output_folder,'temp/',base_name+str(sequence_idx)+'_stablized.npy')):
        imgs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 3] Stabilize the expression In Video:"):            
            source_img = trans_image(frames_pil[idx]).unsqueeze(0).to(device)
            semantic_source_numpy = semantic_npy[idx:idx+1]
            ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
            coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(device)        
            # hacking the new expression
            coeff[:, :64, :] = expression[None, :64, None].to(device) 
            with torch.no_grad():
                output = D_Net(source_img, coeff)
            img_stablized = np.uint8((output['fake_image'].squeeze(0).permute(1,2,0).cpu().clamp_(-1, 1).numpy() + 1 )/2. * 255)
            imgs.append(cv2.cvtColor(img_stablized,cv2.COLOR_RGB2BGR)) 
        np.save(os.path.join(output_folder,'temp/',base_name+str(sequence_idx)+'_stablized.npy'),imgs)
        del D_Net
    else:
        print('[Step 3] Using saved stabilized video.')
        imgs = np.load(os.path.join(output_folder,'temp/',base_name+str(sequence_idx)+'_stablized.npy'))
    torch.cuda.empty_cache()
    new_audio_path = os.path.join(output_folder,base_name,base_name+"_"+str(sequence_idx)+"_audio.wav")    
    segment_audio(audio_path,sequence[0]/fps,(sequence[-1]+1)/fps,new_audio_path)        
    wav = audio.load_wav(new_audio_path, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80./fps, 0, []
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("[Step 4] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
    imgs = imgs[:len(mel_chunks)]
    full_frames = full_frames[:len(mel_chunks)]  
    lm = lm[:len(mel_chunks)]
    
    imgs_enhanced = []
    for idx in tqdm(range(len(imgs)), desc='[Step 5] Reference Enhancement'):
        img = imgs[idx]
        pred, _, _ = enhancer.process(img, img, face_enhance=True, possion_blending=False)
        imgs_enhanced.append(pred)
    gen = datagen(imgs_enhanced.copy(), mel_chunks, full_frames, None,asd_coordinates,output_folder,base_name)

    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('{}/{}/{}_{}_noaudio.mp4'.format(output_folder,base_name,base_name,str(sequence_idx)), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
    
    if lipsync_options.up_face != 'original':
        instance = GANimationModel()
        instance.initialize()
        instance.setup()

    kp_extractor = KeypointExtractor()
    for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(tqdm(gen, desc='[Step 6] Lip Synthesis:', total=int(np.ceil(float(len(mel_chunks)) / lipsync_options.LNet_batch_size)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        img_original = torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(device)/255. # BGR -> RGB
        
        with torch.no_grad():
            incomplete, reference = torch.split(img_batch, 3, dim=1) 
            pred, low_res = model(mel_batch, img_batch, reference)
            pred = torch.clamp(pred, 0, 1)

            if lipsync_options.up_face in ['sad', 'angry', 'surprise']:
                tar_aus = exp_aus_dict[lipsync_options.up_face]
            else:
                pass
            
            if lipsync_options.up_face == 'original':
                cur_gen_faces = img_original
            else:
                test_batch = {'src_img': torch.nn.functional.interpolate((img_original * 2 - 1), size=(128, 128), mode='bilinear'), 
                              'tar_aus': tar_aus.repeat(len(incomplete), 1)}
                instance.feed_batch(test_batch)
                instance.forward()
                cur_gen_faces = torch.nn.functional.interpolate(instance.fake_img / 2. + 0.5, size=(384, 384), mode='bilinear')
                
            if lipsync_options.without_rl1 is not False:
                incomplete, reference = torch.split(img_batch, 3, dim=1)
                mask = torch.where(incomplete==0, torch.ones_like(incomplete), torch.zeros_like(incomplete)) 
                pred = pred * mask + cur_gen_faces * (1 - mask) 
        
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        torch.cuda.empty_cache()
        for p, f, xf, c in zip(pred, frames, f_frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            ff = xf.copy() 
            ff[y1:y2, x1:x2] = p
            
            # month region enhancement by GFPGAN
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                ff, has_aligned=False, only_center_face=True, paste_back=True)
                # 0,   1,   2,   3,   4,   5,   6,   7,   8,  9, 10,  11,  12,
            mm = [0,   0,   0,   0,   0,   0,   0,   0,   0,  0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
            mouse_mask = np.zeros_like(restored_img)
            tmp_mask = enhancer.faceparser.process(restored_img[y1:y2, x1:x2], mm)[0]
            mouse_mask[y1:y2, x1:x2]= cv2.resize(tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.

            height, width = ff.shape[:2]
            restored_img, ff, full_mask = [cv2.resize(x, (512, 512)) for x in (restored_img, ff, np.float32(mouse_mask))]
            img = Laplacian_Pyramid_Blending_with_mask(restored_img, ff, full_mask[:, :, 0], 10)
            pp = np.uint8(cv2.resize(np.clip(img, 0 ,255), (width, height)))

            pp, orig_faces, enhanced_faces = enhancer.process(pp, xf, bbox=c, face_enhance=False, possion_blending=True)
            out.write(pp)
    out.release()
    
    seq_outfile = outfile.split(".")[0]+f"_{str(sequence_idx)}"+"."+outfile.split(".")[-1]
    if not os.path.isdir(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile), exist_ok=True)        
    command = 'ffmpeg -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, '{}/{}/{}_{}_noaudio.mp4'.format(output_folder,base_name,base_name,str(sequence_idx)), seq_outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')
    print('seq_outfile:', seq_outfile)

def merge_audio_video(v_noaudio_path,portion_audio_file,output_video_path):
    
    # Combine video without audio with the extracted audio
    ffmpeg_command_combine = [
        'ffmpeg',
        '-y',
        '-i', v_noaudio_path,  # Input video
        '-i', portion_audio_file,  # Input audio
        '-c:v', 'copy',  # Copy video without re-encoding
        '-c:a', 'aac',  # Use AAC codec for audio
        '-strict', 'experimental',
        output_video_path  # Output video file
    ]
    
    subprocess.run(ffmpeg_command_combine)
    print(f"Final video saved with audio to {output_video_path}")

def segment_audio(audio_file,start_time,end_time,portion_audio_file):
    ffmpeg_command_extract_audio = [
        'ffmpeg',
        '-loglevel', "error",
        '-y',  # Overwrite existing files
        '-i', audio_file,  # Input audio file
        '-ss', str(start_time),  # Start time of audio
        '-to', str(end_time),  # End time of audio
        '-c', 'copy',  # Copy without re-encoding
        portion_audio_file  # Output temp audio file
    ]

    subprocess.run(ffmpeg_command_extract_audio)
    print(f"Extracted audio from {start_time} to {end_time} seconds.")

def create_video_from_frames(full_frames,v_noaudio_path,fps):
    
    if len(full_frames) == 0:
        raise ValueError("The frames list is empty.")

    # Convert all frames to numpy arrays (if they're PIL Images)
    frames_np = [np.array(frame) if isinstance(frame, Image.Image) else frame for frame in full_frames]

    # Get the size of the frame (assuming all frames are the same size)
    frame_height, frame_width, _ = frames_np[0].shape

    # Define the codec and create VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec as needed
    video_writer = cv2.VideoWriter(v_noaudio_path, fourcc, fps, (frame_width, frame_height))

    # Write each frame to the video
    for frame in frames_np:
        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()
    print(f"Video without audio saved to {v_noaudio_path}")

def extract_noface_video(sequence,sequence_idx,full_frames,fps,output_folder,base_name,audio_file):
    v_noaudio_path = os.path.join(output_folder,base_name,base_name+"_"+str(sequence_idx)+"_noaudio.mp4")
    output_video_path  = os.path.join(output_folder,base_name,base_name+"_"+str(sequence_idx)+".mp4")
    portion_audio_file = os.path.join(output_folder,base_name,base_name+"_"+str(sequence_idx)+"_audio.wav")

    start_frame = sequence[0]
    end_frame = sequence[-1]
    start_time = start_frame / fps
    end_time = (end_frame + 1) / fps # Add one to include the last frame duration
    
    create_video_from_frames(full_frames[sequence[0]:sequence[-1]+1],v_noaudio_path,fps)    
    segment_audio(audio_file,start_time,end_time,portion_audio_file)
    merge_audio_video(v_noaudio_path,portion_audio_file,output_video_path)
    """
    # Cleanup temporary files
    os.remove(v_noaudio_path)
    os.remove(portion_audio_file)
    """

def create_output_video(outfile,videos_folder,base_name):
    #'{}/temp/{}_{}.mp4'.format(output_folder,base_name,str(sequence_idx))
    # Concatenate all sequence videos into one final video
    with open("videos_list.txt", "w") as f:
        video_list=[]
        for video_name in os.listdir(videos_folder):
            if ((base_name+"_") in video_name) and not("_audio" in video_name):
                video_list.append(f"file '{os.path.join(videos_folder,video_name)}'\n")
        video_list = sorted(video_list)
        f.write("".join(video_list))
    # Final concatenation command
    ffmpeg_concat_command = [
        'ffmpeg', 
        '-y', 
        '-f', 'concat', 
        '-safe', '0', 
        '-i', 'videos_list.txt', 
        '-c', 'copy', 
        outfile
    ]
    
    subprocess.run(ffmpeg_concat_command)
    """
    # Clean up temporary files
    for video in sequence_videos:
        os.remove(video)
    """
