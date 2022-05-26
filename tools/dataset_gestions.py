import pandas as pd
import os

def refactor_id_frames_extractor():
    """
    This function deletes the frames (goten by 'frame extractor') that are not labeled.
    """

    # path to db
    path = './data/Apple_Tracking_db'

    # read excel that contains all the information about the frames
    df = pd.read_excel(os.path.join(path, 'db_info.xlsx'))

    # Important info from the excel
    videos_names = df['Video'].unique()

    for idx, video_name in enumerate(videos_names):
        if str(video_name) != 'nan':
            path_to_frames = os.path.join(path, video_name, 'images')
            path_to_segmentation = os.path.join(path, video_name, 'segmentation')
            # if frames are extracted and labeling is done
            if os.path.exists(path_to_segmentation) and os.path.exists(path_to_frames):
                # get row index from the df
                row_idx = df[df['Video'] == video_name].index.values[0]
                # the initial frame on which the extractor starts
                frame_init = df.loc[row_idx, 'Frame inicial extractor frames']
                # number of frames labeled
                nframes = df.loc[row_idx, 'NFrames'] - 1  # -1: apaño por que he cambiado una cosa...!
                # get the path files inside the folder
                path_frames = os.listdir(path_to_frames)

                # only refactor when the number of frames labeled is different from the number of frames extracted.
                # we divide by 3 because we have 3 channels, color, depth and IR
                if len(path_frames)/3 != nframes + 1:
                    print(f'Refactoring frames from => {video_name}...')
                    for path_frame in path_frames:
                        # split the string with _ amd get the number of the frame
                        num = path_frame.split('_')[-2]
                        if int(num) < int(frame_init) or int(num) > int(frame_init + nframes):
                            # remove the file
                            os.remove(os.path.join(path_to_frames, path_frame))


if __name__ == "__main__":
    refactor_id_frames_extractor()