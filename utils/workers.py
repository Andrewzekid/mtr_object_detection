#    utils/workers.py - Background-thread workers for long-running tasks.
#
#    USAGE (TaskWorker):
#        from utils.workers import TaskWorker
#
#        def my_long_task(progress_callback=None,
#                         status_callback=None,
#                         log_callback=None,
#                         is_cancelled=None):
#            for i in range(100):
#                if is_cancelled and is_cancelled():
#                    return {"success": False, "cancelled": True}
#                progress_callback(i)
#                status_callback(f"Step {i}/100")
#                log_callback(f"did step {i}")
#            return {"success": True, "result": "done"}
#
#        worker = TaskWorker(my_long_task)
#        worker.progress.connect(my_progress_bar.setValue)
#        worker.status.connect(my_status_label.setText)
#        worker.log.connect(my_log_view.append)
#        worker.finished.connect(on_done)
#        worker.error.connect(on_error)
#        worker.start()
#
#        worker.cancel()   # cooperative cancellation from another thread
#
#    USAGE (BatchTaskWorker):
#        from utils.workers import BatchTaskWorker
#        batch = BatchTaskWorker(items=image_paths, process_func=run_inference)
#        batch.progress.connect(...)
#        batch.start()
#
#    RUN AS A ONE-LINER:
#        python -c "from utils.workers import TaskWorker; print('OK')"
#
#    SIGNALS (TaskWorker / BatchTaskWorker):
#        progress(int)            - 0..100
#        status(str)              - human-readable status message
#        finished(object)         - task return value
#        error(str)               - stack trace + message on exception
#        log(str)                 - (TaskWorker only) free-form log line
#        item_completed(int, obj) - (BatchTaskWorker only) per-item result
#
#    REQUIREMENTS:
#        pip install PyQt6

"""
Long-running tasks wrapper using QThread for asynchronous execution.
Provides progress signals and callbacks for status updates.
"""

from PyQt6.QtCore import QThread, pyqtSignal
from typing import Callable, Any, Optional
import traceback


class TaskWorker(QThread):
    """Generic worker for running long tasks in background threads."""
    
    # Signals
    progress = pyqtSignal(int)  # Progress percentage (0-100)
    status = pyqtSignal(str)    # Status message
    finished = pyqtSignal(object)  # Result object
    error = pyqtSignal(str)     # Error message
    log = pyqtSignal(str)       # Log message
    
    def __init__(self, task_func: Callable, *args, **kwargs):
        """
        Initialize the task worker.
        
        Args:
            task_func: The function to execute in the background
            *args: Positional arguments for the task function
            **kwargs: Keyword arguments for the task function
        """
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self._is_cancelled = False
    
    def run(self):
        """Execute the task and emit signals for progress/status."""
        try:
            # Add progress callback to kwargs
            self.kwargs['progress_callback'] = self._progress_callback
            self.kwargs['status_callback'] = self._status_callback
            self.kwargs['log_callback'] = self._log_callback
            self.kwargs['is_cancelled'] = self._is_cancelled_check
            
            # Execute the task
            result = self.task_func(*self.args, **self.kwargs)
            
            if not self._is_cancelled:
                self.finished.emit(result)
            else:
                self.status.emit("Task cancelled")
                
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
    
    def _progress_callback(self, value: int):
        """Emit progress signal."""
        self.progress.emit(value)
    
    def _status_callback(self, message: str):
        """Emit status signal."""
        self.status.emit(message)
    
    def _log_callback(self, message: str):
        """Emit log signal."""
        self.log.emit(message)
    
    def _is_cancelled_check(self) -> bool:
        """Check if task has been cancelled."""
        return self._is_cancelled
    
    def cancel(self):
        """Request task cancellation."""
        self._is_cancelled = True


class ProgressTracker:
    """Helper class to track and report progress of multi-step operations."""
    
    def __init__(self, total_steps: int, progress_callback: Optional[Callable[[int], None]] = None):
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_callback = progress_callback
    
    def step(self, message: str = ""):
        """Advance one step and report progress."""
        self.current_step += 1
        percentage = int((self.current_step / self.total_steps) * 100)
        if self.progress_callback:
            self.progress_callback(percentage)
        return message
    
    def reset(self):
        """Reset the progress tracker."""
        self.current_step = 0


class BatchTaskWorker(QThread):
    """Worker for processing batches of items with progress tracking."""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    item_completed = pyqtSignal(int, object)  # index, result
    finished = pyqtSignal(list)  # All results
    error = pyqtSignal(str)
    
    def __init__(self, items: list, process_func: Callable, *args, **kwargs):
        """
        Initialize batch worker.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            *args, **kwargs: Additional arguments for process_func
        """
        super().__init__()
        self.items = items
        self.process_func = process_func
        self.args = args
        self.kwargs = kwargs
        self.results = []
        self._is_cancelled = False
    
    def run(self):
        """Process all items in the batch."""
        self.results = []
        total = len(self.items)
        
        for i, item in enumerate(self.items):
            if self._is_cancelled:
                self.status.emit("Batch processing cancelled")
                return
            
            try:
                result = self.process_func(item, *self.args, **self.kwargs)
                self.results.append(result)
                self.item_completed.emit(i, result)
            except Exception as e:
                self.error.emit(f"Error processing item {i}: {str(e)}")
                self.results.append(None)
            
            # Update progress
            progress = int(((i + 1) / total) * 100)
            self.progress.emit(progress)
            self.status.emit(f"Processing {i + 1}/{total}")
        
        self.finished.emit(self.results)
    
    def cancel(self):
        """Cancel batch processing."""
        self._is_cancelled = True