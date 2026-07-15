project_root/
│
├── app.py                      # Main entry point (initializes GUI and page router)
├── core/                       # Functional backend scripts (headless)
│   ├── __init__.py
│   ├── dataset_creator.py      # Undistort, random selection, train/test split
│   ├── data_processor.py       # Augmentation, dataset statistics
│   ├── model_trainer.py        # YOLO training pipeline
│   ├── model_evaluator.py      # Evaluation on unseen data, GT comparison
│   ├── visualizer.py           # Generate prediction folders and timeline visualizations
│   └── models_inference.py     # API/Local wrappers for SAM3 and Qwen3.6
│
├── gui/                        # UI Components
│   ├── __init__.py
│   ├── main_window.py          # Sidebar/Main menu navigation and layout wrapper
│   └── pages/                  # Individual workflow & LLM pages
│       ├── __init__.py
│       ├── main_menu_page.py   # Dashboard overview
│       ├── workflow_pages.py   # Steps 1-6 (Labeling, Augment, Split, Train, Eval, Viz)
│       ├── qwen_page.py        # Qwen3.6 Inference UI
│       └── sam_page.py         # SAM3 Segmentation UI
│
└── utils/                      # Helper modules (config, logger, thread workers)
    ├── __init__.py
    ├── config.py               # Global settings and path management
    └── workers.py              # Long-running tasks wrapper (Threading/QThreads)


1. Functional Core (/core)Each module must be completely decoupled from the GUI, accepting inputs via functions/arguments and returning structured logs or metrics.dataset_creator.pyundistort_camera(input_dir, output_dir, camera_matrix): Fixes wide-angle/metacam distortion.select_random_images(src, dest, count): Sub-samples datasets.split_dataset(src, output_dir, ratios=[0.7, 0.15, 0.15]): Handles Train/Test/Val division.data_processor.pyaugment_dataset(config_dict): Applies flips, rotations, or brightness adjustments.get_statistics(dataset_path): Returns class distributions, image counts, and dimensions.model_trainer.pytrain_yolo(config_path, epochs, batch_size): Spawns the YOLO training loop. Must emit progress logs.model_evaluator.pyevaluate_unseen(model_path, test_data_path): Evaluates model performance.compare_with_gt(predictions, ground_truth): Calculates mAP, Precision, Recall, and Confusion Matrices.models_inference.pyrun_sam3(image_path, bbox, text_prompt) $\rightarrow$ Returns segmented mask overlay.run_qwen(prompt, template_id, output_format) $\rightarrow$ Parses output structured format (e.g., coordinates mapping to JSON/BBoxes).🖥️ UI & Page ArchitectureThe main menu operates as a layout skeleton containing a Sidebar Navigation Menu on the left and a Dynamic Main Content Area on the right.Page Routing Map[Main Window Router]
 ├── 🏠 Main Menu Dashboard (Timeline Progress Overview)
 ├── 🔄 Training Workflow Stack
 │    ├── Page 1: Label Data (Launches external label script/embedded canvas)
 │    ├── Page 2: Data Augmentation & Statistics
 │    ├── Page 3: Train / Test / Val Split
 │    ├── Page 4: Train Model (YOLO UI settings + progress bars)
 │    ├── Page 5: Evaluate Model (Metrics, charts, and GT comparisons)
 │    └── Page 6: Timeline Visualizations (Chronological run output browser)
 ├── 🤖 Qwen3.6 Multi-modal Tool
 └── 🧬 SAM3 Vision Tool

You are tasked with generating a desktop application (e.g., Python using PyQt6/PySide6 or Tkinter) based on the above architecture. Please adhere strictly to the following implementation constraints:Asynchronous Execution (Threading): All functions from core/ (especially training, augmentation, and LLM inference) must run on separate background threads. Do not freeze the GUI main thread. Use a worker system with progress bars and signals/callbacks for status updates.Shared State Management: Maintain a centralized configurations/state manager class instance (utils/config.py) that acts as a single source of truth for variables like current_dataset_path, trained_model_path, and pipeline progress states.UI Page Layouts:Main Dashboard: Implement a visual timeline widget showing the 6 workflow steps. Clicking a step jumps to that specific page.Qwen Page: Provide a dropdown menu for template prompts, a text field for custom overrides, a format selector (e.g., JSON, YAML, Bounding Box), and an output preview window.SAM3 Page: Provide an canvas to input image/bounding boxes, text inputs, and a display canvas showcasing the output masked segments.