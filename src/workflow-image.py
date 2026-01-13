

import kfp
import mlrun
from kfp import dsl
from typing import List


###########################
import os
import sys
print(os.getcwd())
print(kfp.__version__)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, os.pardir)
sys.path.insert(0, parent_dir)
parent_dir = os.path.join(parent_dir, os.pardir)
sys.path.insert(0, parent_dir)
print(f'parent dir = {parent_dir}')
###########################

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(name="test-image-in-workflow")
def pipeline(
        amount: int,
        generation_model: str,
        tts_model: str,
        language: str,
        available_voices: List[str],
        min_time: int,
        max_time: int,
        from_date: str,
        to_date: str,
        from_time: str,
        to_time: str,
        num_clients: int,
        num_agents: int,
        generate_clients_and_agents: bool = True,
    ):
    # project = mlrun.get_current_project()

    with dsl.Condition(generate_clients_and_agents == True) as generate_data_condition:

        # Run the function 
        # f = project.get_function("test-image")
        f_run = mlrun.run_function("test-image", handler="handler")

        # Get and Run the structured_data_generator hub function 
        # client_data_generator_function = mlrun.get_function(
        #     "structured_data_generator"
        # )
        client_data_run = mlrun.run_function(
            "structured-data-generator",
            # client_data_generator_function,
            handler="generate_data",
            params={
                "amount": 2,
                "model_name": "gpt-4",
                "language": "en",
                "fields": [
                    f"first_name: in english, no special characters",
                    f"last_name: in english, no special characters",
                    "phone_number",
                    "email",
                    "client_id: no leading zeros, at least 8 digits long, only numbers, this is a primay key field for the database, avoid duplicates as much as possible",
                    "client_city: Enter city, state in the US (e.g., Austin, TX), Not only Texas",
                    "latitude: That correspond to the city",
                    "longitude: That correspond to the city",
                ],
            },
            returns=["clients: file"],
        )
