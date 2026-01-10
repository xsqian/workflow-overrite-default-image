
from kfp import dsl
import mlrun

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(name="test-image-in-workflow")
def pipeline():
    
    project = mlrun.get_current_project()
    
    # Run the ingestion function 
    f = project.get_function("test-image")
    f_run = project.run_function(f)
