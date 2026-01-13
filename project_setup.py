import os
from pathlib import Path
import boto3
import mlrun
import v3io.dataplane
import tempfile



def setup(
    project: mlrun.projects.MlrunProject,
) -> mlrun.projects.MlrunProject:


    # Unpack parameters:
    source = project.get_param(key="source")
    default_image = project.get_param(key="default_image", default=None)
    build_image = project.get_param(key="build_image", default=False)
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base = os.getenv("OPENAI_API_BASE")

    # Set the project git source:
    if source:
        print(f"Project Source: {source}")
        project.set_source(source=source, pull_at_runtime=False)
        

    # Set default image:
    if default_image:
        project.set_default_image(default_image)
        print(f"set deafult image to : {default_image}")

    # Build the image:
    if build_image:
        print("Building default image for the demo:")
        _build_image(project=project,  default_image=default_image)

    # Set the secrets:
    _set_secrets(
        project=project,
        openai_key=openai_key,
        openai_base=openai_base,
    )

    # # Refresh MLRun hub to the most up-to-date version:
    # mlrun.get_run_db().get_hub_catalog(source_name="default", force_refresh=True)

    # Set the functions:
    _set_calls_generation_functions(project=project, image=default_image)   

    # Set the workflows:
    _set_workflows(project=project, image=default_image)

    project.save()

    print("\n", "-"*100)
    print(project.to_yaml())
    print("-"*100, "\n")
    if project.spec.params['default_image']:
        print(f"\nproject default image: {project.spec.params['default_image']}\n")

    return project

def _build_image(project: mlrun.projects.MlrunProject, default_image):
    config = {
        "base_image": "mlrun/mlrun-kfp",
        "torch_index": "https://download.pytorch.org/whl/cpu",
        "onnx_package": "onnxruntime"
    }
    # Define commands in logical groups while maintaining order
    system_commands = [
        # Update apt-get to install ffmpeg (support audio file formats):
        "apt-get update -y && apt-get install ffmpeg -y"
    ]

    infrastructure_requirements = [
        "pip install transformers==4.44.1",
        f"pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url {config['torch_index']}"
    ]

    huggingface_requirements = [
        "pip install bitsandbytes==0.41.1 accelerate==0.24.1 datasets==2.14.6 peft==0.5.0 optimum==1.13.2"
    ]

    other_requirements = [
        "pip install langchain==0.2.17 openai==1.58.1 langchain_community==0.2.19 pydub==0.25.1 streamlit==1.28.0 st-annotated-text==4.0.1 spacy==3.7.1 librosa==0.10.1 presidio-anonymizer==2.2.34 presidio-analyzer==2.2.34 nltk==3.8.1 flair==0.13.0 htbuilder==0.6.2",
        "pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1.tar.gz",
        # "python -m spacy download en_core_web_lg",

        "pip install SQLAlchemy==2.0.31 pymysql requests_toolbelt==0.10.1",
        "pip uninstall -y onnxruntime-gpu onnxruntime",
        f"pip install {config['onnx_package']}",
        "pip uninstall -y protobuf",
        "pip install protobuf"
    ]    

    # Combine commands in the required order
    commands = (
            system_commands +
            infrastructure_requirements +
            huggingface_requirements +
            other_requirements
    )

    # Build the image
    assert project.build_image(
        image = default_image,
        base_image="mlrun/mlrun-kfp",
        commands=commands,
        set_as_default=True,
        overwrite_build_params=True
    )
    
def _set_secrets(
    project: mlrun.projects.MlrunProject,
    openai_key: str,
    openai_base: str,

):
    # Must have secrets:
    assert openai_key and openai_base, "openai_key and openai_base must be set"
    project.set_secrets(
        secrets={
            "OPENAI_API_KEY": openai_key,
            "OPENAI_API_BASE": openai_base,

        }
    )

def _set_function(
        project: mlrun.projects.MlrunProject,
        func: str,
        name: str,
        kind: str,
        with_repo: bool = None,
        image: str = None,
        apply_auto_mount: bool = True,
):
    # Set the given function:
    print(f"name: {name}")
    print(f"with_repo: {with_repo}")
    if with_repo is None:
        with_repo =  not func.startswith("hub://")
        
    mlrun_function = project.set_function(
        func=func, name=name, kind=kind, with_repo=with_repo, image=image,
    )


    # Save:
    mlrun_function.save()
    # project.set_function(f"db://{project.name}/{name}", name=name)


def _set_calls_generation_functions(
    project: mlrun.projects.MlrunProject,
    image: str = ""
):
    # Client and agent data generator
    _set_function(
        project=project,
        func="hub://structured_data_generator",
        name="structured-data-generator",
        kind="job",
        image=image,
        with_repo=False,
        apply_auto_mount=True,
    )
    
    
    # Conversation generator:
    _set_function(
        project=project,
        func="./src/test-image.py",
        name="test-image",
        kind="job",
        image=image,
        with_repo=False,
        apply_auto_mount=True,
    )


def _set_workflows(project: mlrun.projects.MlrunProject, image):

    project.set_workflow(
        name="test-image-in-workflow", workflow_path="src/workflow-image.py",image=image, 
    )
