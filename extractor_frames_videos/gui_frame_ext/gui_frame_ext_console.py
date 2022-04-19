"""
Project: Fruit size estimation
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: November 2021
Description:

User interface that contains functions related to MAIN SCREEN.

Use:
    ui_frame_extractor_config = GUIFrameExtractorConfig(ui_path_config_file)
    app = GUIFrameExtractorConsole(ui_frame_extractor_config, dataset_manager_config_obj, frames_extractor_config)
    app.mainloop()y
"""
import os
import tkinter as tk
from tkinter import filedialog
from gui_frame_ext.about_window import AboutWindow

from helpers.helper_validation import digit_validation
from dataset_management.dataset_manager import DatasetConfig
from dataset_management.dataset_manager import DatasetManager
from video_extraction_management.frame_extraction import FramesVideoManager


class GUIFrameExtractorConsole(tk.Tk):
    r_config = None
    dataset_config = None
    frames_extractor_config = None

    def __init__(self, r_config, dataset_config, frames_extractor_config, master=None):
        super().__init__(master)
        # ---------------------------
        # configuration parameters
        self.r_config = r_config  # assign config
        self.dataset_config = dataset_config
        self.frames_extractor_config = frames_extractor_config
        # ---------------------------

        self.geometry(self.r_config.geometry_main)
        self.title(r_config.app_title)
        self.resizable(width=False, height=False)  # do not change the size
        self.attributes('-topmost', True)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.createWidgets()
        self.createMenuBars()

    def createWidgets(self):
        self.left_frame = tk.Frame(self)
        self.left_frame.grid(row=0, column=0)

        self.extract_folder_data_frame = tk.Frame(self)
        self.extract_folder_data_frame.grid(row=1, column=0)

        self.extract_data_frame = tk.Frame(self)
        self.extract_data_frame.grid(row=2, column=0)
        self.labelled_data_frame = tk.Frame(self)
        self.labelled_data_frame.grid(row=3, column=0)
        self.message_frame = tk.Frame(self)
        self.message_frame.grid(row=4, column=0)
        self.control_bar_frame = tk.Frame(self)
        self.control_bar_frame.grid(row=5, column=0)

        ############### CREATE HIERARCHY ######################
        self.user_path_label = tk.Label(self.left_frame, text='User path:')
        self.user_path_label.grid(row=0, column=0, sticky=tk.EW)
        self.user_path_entry = tk.Entry(self.left_frame)
        self.user_path_entry.grid(row=0, column=1, sticky=tk.EW)
        self.user_path_entry.insert(0, self.r_config.user_path)
        # --------------
        self.dataset_name_label = tk.Label(self.left_frame, text='Dataset name:')
        self.dataset_name_label.grid(row=1, column=0, sticky=tk.EW)
        # -----------------------
        self.dataset_name_entry = tk.Entry(self.left_frame)
        self.dataset_name_entry.grid(row=1, column=1, sticky=tk.EW)
        self.dataset_name_entry.insert(0, self.dataset_config.dataset_name)

        # --------------
        self.base_path_label = tk.Label(self.left_frame, text='Base path:')
        self.base_path_label.grid(row=2, column=0, sticky=tk.EW)
        # -----------------------
        self.base_path_entry = tk.Entry(self.left_frame)
        self.base_path_entry.grid(row=2, column=1, sticky=tk.EW)
        # --------------
        # insert default values
        self.create_dataset_button = tk.Button(self.left_frame, text='Create dataset hierarchy',
                                               command=self.create_dataset_folder)
        self.create_dataset_button.grid(row=4, column=0, sticky=tk.EW)
        ########################################################
        # extract from a folder
        self.an_input_folder_label = tk.Label(self.extract_folder_data_frame, text='Selected video folder:')
        self.an_input_folder_label.grid(row=0, column=0, sticky=tk.EW)
        self.an_input_folder_entry = tk.Entry(self.extract_folder_data_frame)
        self.an_input_folder_entry.grid(row=0, column=1, sticky=tk.EW)

        self.select_folder_button = tk.Button(self.extract_folder_data_frame, text='Select folder',
                                              command=self.select_folder_data)
        self.select_folder_button.grid(row=5, column=0, sticky=tk.EW)
        self.load_folder_button = tk.Button(self.extract_folder_data_frame, text='Generate dataset from folder',
                                            command=self.extract_folder_data)
        self.load_folder_button.grid(row=5, column=1, sticky=tk.EW)

        ########################################################
        # extract from one file
        self.an_input_file_label = tk.Label(self.extract_data_frame, text='Selected video file:')
        self.an_input_file_label.grid(row=0, column=0, sticky=tk.EW)
        # # -----------------------
        self.an_input_file_entry = tk.Entry(self.extract_data_frame)
        self.an_input_file_entry.grid(row=0, column=1, sticky=tk.EW)

        self.an_offset_label = tk.Label(self.extract_data_frame, text='Offset to start in seconds:')
        self.an_offset_label.grid(row=1, column=0, sticky=tk.EW)
        # # -----------------------
        self.an_offset_entry = tk.Entry(self.extract_data_frame, validate="key")
        self.an_offset_entry.grid(row=1, column=1, sticky=tk.EW)
        self.an_offset_entry['validatecommand'] = (self.an_offset_entry.register(digit_validation), '%P', '%d')
        # # --------------
        self.a_number_of_frames_label = tk.Label(self.extract_data_frame, text='Number of frames:')
        self.a_number_of_frames_label.grid(row=2, column=0, sticky=tk.EW)
        # # -----------------------
        self.a_number_of_frames_entry = tk.Entry(self.extract_data_frame, validate="key")
        self.a_number_of_frames_entry.grid(row=2, column=1, sticky=tk.EW)
        self.a_number_of_frames_entry['validatecommand'] = (
            self.a_number_of_frames_entry.register(digit_validation), '%P', '%d')
        # # --------------
        # # insert default values
        self.an_offset_entry.insert(1, "1")  # todo: this may be loaded from a configuration file
        self.a_number_of_frames_entry.insert(1, "1")  # todo: this may be loaded from a configuration file

        self.select_file_button = tk.Button(self.extract_data_frame, text='Select file', command=self.select_file_data)
        self.select_file_button.grid(row=5, column=0, sticky=tk.EW)

        self.load_file_button = tk.Button(self.extract_data_frame, text='Generate dataset from file',
                                          command=self.extract_file_data)
        self.load_file_button.grid(row=5, column=1, sticky=tk.EW)

        self.migrate_labelled_data_button = tk.Button(self.labelled_data_frame, text='Migrate labelled data',
                                                      command=self.not_implemented_yet)
        self.migrate_labelled_data_button.grid(row=0, column=0, sticky=tk.EW)

        self.quitButton = tk.Button(self.control_bar_frame, text='Quit', command=self.quit_app)
        self.quitButton.grid(row=0, column=0, sticky=tk.EW)

        ################
        self.messages_label = tk.Label(self.message_frame, text='Messages:')
        self.messages_label.grid(row=0, column=0, sticky=tk.EW)
        self.messages_info = tk.Text(self.message_frame, width=40, height=5)
        self.messages_info.grid(row=1, column=0, sticky=tk.EW)
        self.results_info = tk.Text(self.message_frame, width=10, height=5)
        self.results_info.grid(row=11, column=0, sticky=tk.EW)

    def createMenuBars(self):
        """
        Add menues to the UI
        :return:
        """
        self.menubar = tk.Menu(self)

        self.menu_file = tk.Menu(self.menubar, tearoff=False)  # delete dash lines
        self.menu_file.add_command(label="A_command", command=self.not_implemented_yet)
        self.menu_file.add_command(label="Quit", command=self.quit_app)

        self.menu_help = tk.Menu(self.menubar, tearoff=False)  # delete dash lines
        self.menu_help.add_command(label="Help...", command=self.open_about_data)
        self.menu_help.add_command(label="About...", command=self.open_about_data)

        self.menubar.add_cascade(menu=self.menu_file, label='File', underline=0)
        self.menubar.add_cascade(menu=self.menu_help, label='About', underline=0)
        self.config(menu=self.menubar)  # add menu to window

    def open_about_data(self):
        about_windows = AboutWindow(self)
        about_windows.grab_set()

    def not_implemented_yet(self):
        print("Not implemented yet!!!")

#    def clean_temporal_files(self):
#        self.messages_info.delete("1.0", "end")
#        self.results_info.delete("1.0", "end")
#        print("Remove files")
#        # remove_files(self.r_config.output_split_folder, self.r_config.file_type)
#        # remove_files(self.r_config.output_normalized_folder, self.r_config.file_type)

    def clean_text_widgets(self):
        self.messages_info.delete("1.0", "end")
        self.results_info.delete("1.0", "end")

    def create_dataset_folder(self):
        self.clean_text_widgets
        analyze_status_str = ""
        directory_selected = filedialog.askdirectory(initialdir=self.r_config.input_dataset_folder)

        if directory_selected == ():
            analyze_status_str = "A directory has not been selected " + "\n"
            # TODO: 03/03/2022 change this if sentence
        else:
            print("Directory selected!!!")
            self.base_path_entry.delete(0, "end")
            self.base_path_entry.insert(0, directory_selected)

            base_path = os.path.join(self.base_path_entry.get())
            dataset_name = self.dataset_name_entry.get()

            print("base_path -->", base_path)
            print("dataset_name -->", dataset_name)
            self.dataset_config = DatasetConfig(base_path, dataset_name)
            dataset_manager_obj = DatasetManager(self.dataset_config)
            dataset_manager_obj.create_hierarchy()
            print("CREATING DATASET HIERARCHY!!!")

        analyze_status_str = f"Created at {directory_selected}" + "\n"
        self.messages_info.insert("1.0", analyze_status_str)

    def select_folder_data(self):
        analyze_status_str = ""
        # todo: check this initial folder
        # todo:; put in variables extension settings
        directory_selected = filedialog.askdirectory(initialdir=self.r_config.input_dataset_folder)

        if directory_selected == "":
            analyze_status_str = "A directory has not been selected " + "\n"
            print(analyze_status_str)
        else:
            print('directory_selected -->', directory_selected)
            self.an_input_folder_entry.delete(0, "end")
            self.an_input_folder_entry.insert(0, os.path.join(directory_selected))
        # ----------------------------------------
        analyze_status_str = directory_selected
        self.messages_info.insert("1.0", analyze_status_str)
        # ----------------------------------------

    def extract_folder_data(self):
        # self.clean_text_widgets
        analyze_status_str = ""
        results_info_str = ""
        # todo: check this initial folder
        # todo:; put in variables extension settings
        directory_selected = os.path.join(self.an_input_folder_entry.get())

        if directory_selected == "":
            analyze_status_str = "A directory has not been selected " + "\n"
        else:
            print(directory_selected)
            print('MAKING SOME STUFF HERE! processing a folder with videos')
            # todo: update automatic of limit offset
            # todo: update number of frames
            self.an_input_folder_entry.delete(0, "end")
            self.an_input_folder_entry.insert(0, directory_selected)
            ##################################
            an_offset = int(self.an_offset_entry.get())
            a_number_of_frames = int(self.a_number_of_frames_entry.get())
            print('an_offset -->', an_offset)
            print('a_number_of_frames -->', a_number_of_frames)
            #################################
            print('directory_selected -->', directory_selected)
            #################################
            #directory_selected = os.path.join('C:\\', 'recorded_video')
            # TODO: 03/03/2022 CHECK DIRECTORY_SELECTED

            self.frames_extractor_config.path_images_ouput = self.dataset_config.dataset_images_path
            self.frames_extractor_config.path_annotations_ouput = self.dataset_config.dataset_annotations_path  # TODO: 15/02/2022 temporal
            track_file = os.path.join(self.dataset_config.dataset_sets_path, 'all.txt')

            for a_filename in os.listdir(directory_selected):
                if a_filename.endswith(self.r_config.file_extension_to_search):
                    a_path_filename = directory_selected+'\\'+a_filename
                    print(a_filename)
                    print(a_path_filename)
                    frames_extractor_obj = FramesVideoManager(self.frames_extractor_config, a_path_filename)
                    [frames_written, errors, output_folder] = frames_extractor_obj.export_frames_to_files(track_file, an_offset, a_number_of_frames)
                    results_info_str = f"frames written {frames_written} errors {errors} output folder {output_folder}"
                    #break
                    #########################
        # ----------------------------------------
        analyze_status_str = directory_selected
        self.messages_info.insert("1.0", analyze_status_str)
        self.results_info.insert("1.0", results_info_str)

    def select_file_data(self):
        # self.clean_text_widgets
        analyze_status_str = ""
        results_info_str = ""
        # todo: check this initial folder
        # todo:; put in variables extension settings
        path_filename_selected = filedialog.askopenfilename(initialdir=self.r_config.file_browser_input_folder,
                                                            title="Select a File", filetypes=(
                ("Text files", self.r_config.file_extension_to_search), ("all files", "*.mkv")))

        if path_filename_selected == "":
            analyze_status_str = "A file has not been selected " + "\n"
        else:
            # todo: update automatic of limit offset, update number of frames
            self.an_input_file_entry.delete(0, "end")
            self.an_input_file_entry.insert(0, path_filename_selected)
            #################################
            an_input_file = os.path.join(path_filename_selected)
            an_offset = int(self.an_offset_entry.get())
            a_number_of_frames = int(self.a_number_of_frames_entry.get())
            #################################
            print('path_filename_selected -->', path_filename_selected)
            print('an_input_file -->', an_input_file)
            print('an_offset -->', an_offset)
            print('a_number_of_frames -->', a_number_of_frames)
            #################################
        # ----------------------------------------
        analyze_status_str = path_filename_selected
        self.messages_info.insert("1.0", analyze_status_str)

    def extract_file_data(self):
        analyze_status_str = ""
        results_info_str = ""
        path_filename_selected = os.path.join(self.an_input_file_entry.get())

        if path_filename_selected == "":
            analyze_status_str = "A file has not been selected " + "\n"
        else:
            print(path_filename_selected)
            an_input_file = os.path.join(path_filename_selected)
            an_offset = int(self.an_offset_entry.get())
            a_number_of_frames = int(self.a_number_of_frames_entry.get())
            print('an_input_file -->', an_input_file)
            print('an_offset -->', an_offset)
            print('a_number_of_frames -->', a_number_of_frames)
            self.frames_extractor_config.path_images_ouput = self.dataset_config.dataset_images_path
            self.frames_extractor_config.path_annotations_ouput = self.dataset_config.dataset_annotations_path  # 15/02/2022 temporal
            track_file = os.path.join(self.dataset_config.dataset_sets_path, 'all.txt')
            #########################
            frames_extractor_obj = FramesVideoManager(self.frames_extractor_config, an_input_file)
            [frames_written, errors, output_folder] = frames_extractor_obj.export_frames_to_files(track_file, an_offset,
                                                                                                  a_number_of_frames)
            results_info_str = f"frames written {frames_written} errors {errors} output folder {output_folder}"
        # ----------------------------------------
        analyze_status_str = path_filename_selected
        self.messages_info.insert("1.0", analyze_status_str)
        self.results_info.insert("1.0", results_info_str)

    def quit_app(self):
        # ---------------------------------------------
        self.quit
        self.destroy()

# todo: add internacionalization
