
from kfp import dsl
import mlrun

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(name="breast-cancer-demo")
def pipeline(model_name="cancer-classifier"):
    
    project = mlrun.get_current_project()
    
    # Run the ingestion function with the new image and params
    ingest = mlrun.run_function(
        "data-prep",
        name="get-data",
        returns=["dataset", "label_column"],
    )

    client_data_generator_function = project.get_function(
        "structured_data_generator"
    )
    client_data_run = project.run_function(
        client_data_generator_function,
        handler="generate_data",
        name="client-data-generator",
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
