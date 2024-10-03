import streamlit as st

# Define paths to the video and audio files
original_video_path = r"./input/multiface.mp4"
translated_audio_path = r"./input/multiface.wav"
lipsync_video_3_path = r"./output/output.mp4"
lipsync_video_1_path = r"./output/video_out.mp4"
lipsync_video_2_path = r"./output/video_speaking_out.mp4"

def main():
    # Page configuration
    st.set_page_config(layout="wide")

    # First section: Original Video (centered and 50% width)
    st.subheader("Original Video", anchor=None)
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
        st.text("Lipsync output")
        st.video(lipsync_video_3_path)

    # Note: The simultaneous playback button has been removed since JavaScript doesn't work in Streamlit.

if __name__ == "__main__":
    main()
