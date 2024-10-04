import torch
import os,argparse,cv2,sys
import glob
# face enhancement

sys.path.insert(0, 'video_retalking/third_part')
sys.path.insert(0, 'video_retalking/third_part/GPEN')
sys.path.insert(0, 'video_retalking/third_part/GFPGAN')

from video_retalking.third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from video_retalking.third_part.GFPGAN.gfpgan import GFPGANer
# expression control

from video_retalking.utils.inference_utils import  args 
import warnings
warnings.filterwarnings("ignore")

lipsync_options = args

from talknet_asd.demoTalkNet import frames_asd
from utils import find_ordered_sequences_with_status,create_output_video,lipsync,extract_noface_video


def main(video_path,audio_path,output_folder,outfile):    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[Info] Using {} for inference.'.format(device))
    base_name = os.path.basename(video_path).split(".")[0]
    os.makedirs(os.path.join(output_folder,base_name), exist_ok=True)
    os.makedirs(os.path.join(output_folder,"temp"), exist_ok=True)

    enhancer = FaceEnhancement(base_dir='video_retalking/checkpoints', size=512, model='GPEN-BFR-512', use_sr=False, \
                               sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
    restorer = GFPGANer(model_path='video_retalking/checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean', \
                        channel_multiplier=2, bg_upsampler=None)

    asd_output = frames_asd(video_path,output_folder)        
    flist = glob.glob(os.path.join(output_folder,base_name,"pyframes", "*.jpg"))  # Read the frames
    flist.sort()        
    full_frames = [cv2.imread(frame) for frame in flist ]    
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)        
    video_sequences = find_ordered_sequences_with_status(list(range(len(full_frames))),list(asd_output.keys())) # ordred list of sequences (either containing face or not)    
    print(video_sequences)
    
    for sequence_idx,(sequence,contain_face) in enumerate(video_sequences):        
        if contain_face and (len(sequence)>1):
            lipsync(enhancer,restorer,fps,full_frames,asd_output,sequence,sequence_idx,output_folder,base_name,audio_path,outfile,lipsync_options,device)            
        else:            
            extract_noface_video(sequence,sequence_idx,full_frames,fps,output_folder,base_name,audio_path,outfile)           
    create_output_video(outfile,os.path.dirname(outfile),base_name)
    
if __name__=="__main__":    
    parser = argparse.ArgumentParser(description = "Lipsync of a speaker in the presence of many faces in the video")
    parser.add_argument('--videoPath', type=str, help='Full path of input video')
    parser.add_argument('--audioPath', type=str, help='Full path for audio to be used for lipsync')
    parser.add_argument('--outputFolder', type=str, help='Full path for the folder to be used for temporary files and output video')
    parser.add_argument('--outFileName', type=str, help='Name of output lipsync video')
    args = parser.parse_args()
    video_path = args.videoPath
    audio_path = args.audioPath
    output_folder = args.outputFolder
    outfile = os.path.join(output_folder,args.outFileName)
    main(video_path,audio_path,output_folder,outfile)