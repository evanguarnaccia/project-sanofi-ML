import os
import pytest
import wandb
import dataikuapi
import urllib3
from wandb.errors import CommError

# Disable warnings for unverified HTTPS requests (common in internal DSS instances)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_dataiku_wandb_registry_sync():
    """
    Validates that Dataiku Saved Models have corresponding entries 
    in the W&B Model Registry using secrets stored in Dataiku.
    """
    # --- 1. Connection Setup ---
    # These are typically provided by the GitHub Action environment
    url = os.getenv('DATAIKU_INSTANCE_DEV_URL')
    api_key = os.getenv('DATAIKU_API_TOKEN_DEV')
    project_key = os.getenv('DATAIKU_PROJECT_KEY')
    
    if not url:
        pytest.fail("Missing required environment variables: URL")
    if not api_key:
        pytest.fail("Missing required environment variables: API_KEY")
    if not project_key:
        pytest.fail("Missing required environment variables: PROJECT_KEY")

    client = dataikuapi.DSSClient(url, api_key)
    # Ensure client session handles SSL warnings if your DSS uses self-signed certs
    client._session.verify = False 
    
    try:
        project = client.get_project(project_key)
        print(f"\nConnected to Dataiku project: {project_key}")

        # --- 2. Retrieve W&B Secret from Dataiku ---
        auth_info = client.get_auth_info(with_secrets=True)
        secret_value = None
        for secret in auth_info.get("secrets", []):
            if secret.get("key") == "wandbcred":
                secret_value = secret.get("value")
                break
        
        if not secret_value:
            pytest.fail("Secret 'wandbcred' not found in Dataiku user secrets.")

        # --- 3. W&B Authentication ---
        wandb.login(key=secret_value)
        api = wandb.Api()

        # --- 4. Collect Saved Models from Dataiku ---
        saved_model_ids = [sm['id'] for sm in project.list_saved_models()]
        
        if not saved_model_ids:
            print("W&B DEBUG: No saved models found in Dataiku project.")
            return # Test passes as there's nothing to validate

        # --- 5. Collect W&B Model Artifacts ---
        artifacts = []
        for collection in api.registries().collections():
            for artifact in collection.artifacts():
                if artifact.type and artifact.type.lower() == "model":
                    artifacts.append({
                        "artifact": artifact.source_name,
                        "path": artifact.qualified_name
                    })

    except CommError as e:
        pytest.fail(f"Weights & Biases communication error: {e}")
    except Exception as e:
        pytest.fail(f"Setup or connection error: {e}")

    # --- 6. Cross-Reference and Validation ---
    artifact_names = [{'name': a['artifact'], 'path': a['path']} for a in artifacts]
    any_published = False

    for sm in saved_model_ids:
        print(f"\n✅ Checking Dataiku Model: {sm}")

        model = project.get_saved_model(sm)
        try:
            active_version = model.get_active_version()
            active_id = active_version['id']
        except Exception:
            print(f"⚠️ No active version found for model {sm}. Skipping.")
            continue

        model_identifier = f"dataiku-{sm}-{active_id}"
        print(f"Searching W&B for: {model_identifier}")

        # Find matching artifacts
        candidate_artifacts = [a for a in artifact_names if model_identifier in a['name']]

        if not candidate_artifacts:
            print(f"⚠️ No published W&B artifacts found for {model_identifier}")
            continue

        any_published = True
        for art in candidate_artifacts:
            print(f"✅ Match Found: {art['name']}")
            print(f"   Registry Path: {art['path']}")

    # --- 7. Final Assertion ---
    if not any_published and saved_model_ids:
        pytest.fail("Sync Validation Failed: Found models in Dataiku, but none are registered in W&B.")
    
    print("\nVerification Complete.")
