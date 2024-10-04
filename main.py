import streamlit as st

# Define paths to the video and audio files
original_video_path = r"./input/multiface.mp4"

translated_audio_path = r"./input/multiface.wav"

lipsync_video_3_path = r"./output/output.mp4"
lipsync_video_1_path = r"./output/video_out.mp4"
lipsync_video_2_path = r"./output/video_speaking_out.mp4"
lipsync_video_4_path = r"./output/lipsync_bbox2.mp4"

original_video_path2 = r"./input/original.mp4"

translated_audio_path2 = r"./input/original.wav"

lipsync_video_3_path2 = r"./output/output2.mp4"
lipsync_video_1_path2 = r"./output/video_out2.mp4"
lipsync_video_2_path2 = r"./output/video_speaking_out2.mp4"
lipsync_video_4_path2 = r"./output/lipsync_bbox2.mp4"


def main():
    # Page configuration
    st.set_page_config(layout="wide")
    # First section: Original Video (centered and 50% width)
    st.subheader("Original Video (one face per frame)", anchor=None)
    col1, col2, col3 = st.columns([1, 2, 1])  # Center the video by creating 3 columns
    with col2:
        st.video(original_video_path2)  # Let Streamlit control the width

    # Second section: Translated Audio (centered and 50% width)
    st.subheader("Translated Audio", anchor=None)
    col4, col5, col6 = st.columns([1, 2, 1])  # Center the audio using the same approach
    with col5:
        st.audio(translated_audio_path2)

    # Third section: Lipsync Videos with 3 columns (centered)
    st.subheader("Lipsync Videos", anchor=None)
    col7, col8, col9 = st.columns(3)

    with col7:
        st.text("active speaker detection")
        st.video(lipsync_video_1_path2)

    with col8:
        st.text("filtering detected speakers")
        st.video(lipsync_video_2_path2)

    with col9:
        st.text("Lipsync output with asd bboxs")
        st.video(lipsync_video_4_path)

    st.subheader("Lipsync final output", anchor=None)
    col1, col2, col3 = st.columns([1, 2, 1])  # Center the video by creating 3 columns
    with col2:
        st.video(lipsync_video_3_path2)  # Let Streamlit control the width
    
    st.divider()
    st.divider()

    # First section: Original Video (centered and 50% width)
    st.subheader("Original Video (multiple faces per frame)", anchor=None)
    col1, col2, col3 = st.columns([1, 2, 1])  # Center the video by creating 3 columns
    with col2:
        st.video(original_video_path)  # Let Streamlit control the width

    # Second section: Translated Audio (centered and 50% width)
    st.subheader("Translated Audio", anchor=None)
    col4, col5, col6 = st.columns([1, 2, 1])  # Center the audio using the same approach
    with col5:
        st.audio(translated_audio_path)

    # Third section: Lipsync Videos with 3 columns (centered)
    st.subheader("Lipsync Videos", anchor=None)
    col7, col8, col9 = st.columns(3)

    with col7:
        st.text("active speaker detection")
        st.video(lipsync_video_1_path)

    with col8:
        st.text("filtering detected speakers")
        st.video(lipsync_video_2_path)

    with col9:
        st.text("Lipsync output with asd bboxs")
        st.video(lipsync_video_4_path2)

    st.subheader("lipsync final output", anchor=None)
    col1, col2, col3 = st.columns([1, 2, 1])  # Center the video by creating 3 columns
    with col2:
        st.video(lipsync_video_3_path)  # Let Streamlit control the width

if __name__ == "__main__":
    main()
