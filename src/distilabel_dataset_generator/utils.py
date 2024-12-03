import json
import os
from typing import List, Optional, Union

import argilla as rg
import gradio as gr
import numpy as np
import pandas as pd
from gradio.oauth import (
    OAUTH_CLIENT_ID,
    OAUTH_CLIENT_SECRET,
    OAUTH_SCOPES,
    OPENID_PROVIDER_URL,
    get_space,
)
from huggingface_hub import whoami
from jinja2 import Environment, meta

_LOGGED_OUT_CSS = ".main_ui_logged_out{opacity: 0.3; pointer-events: none}"

HF_TOKENS = [os.getenv("HF_TOKEN")] + [os.getenv(f"HF_TOKEN_{i}") for i in range(1, 10)]
HF_TOKENS = [token for token in HF_TOKENS if token]

_CHECK_IF_SPACE_IS_SET = (
    all(
        [
            OAUTH_CLIENT_ID,
            OAUTH_CLIENT_SECRET,
            OAUTH_SCOPES,
            OPENID_PROVIDER_URL,
        ]
    )
    or get_space() is None
)

if _CHECK_IF_SPACE_IS_SET:
    from gradio.oauth import OAuthToken
else:
    OAuthToken = str


def get_login_button():
    return gr.LoginButton(value="Sign in!", size="sm", scale=2).activate()


def get_duplicate_button():
    if get_space() is not None:
        return gr.DuplicateButton(size="lg")


def list_orgs(oauth_token: OAuthToken = None):
    try:
        if oauth_token is None:
            return []
        data = whoami(oauth_token.token)
        if data["auth"]["type"] == "oauth":
            organizations = [data["name"]] + [org["name"] for org in data["orgs"]]
        elif data["auth"]["type"] == "access_token":
            organizations = [org["name"] for org in data["orgs"]]
        else:
            organizations = [
                entry["entity"]["name"]
                for entry in data["auth"]["accessToken"]["fineGrained"]["scoped"]
                if "repo.write" in entry["permissions"]
            ]
            organizations = [org for org in organizations if org != data["name"]]
            organizations = [data["name"]] + organizations
    except Exception as e:
        raise gr.Error(
            f"Failed to get organizations: {e}. See if you are logged and connected: https://huggingface.co/settings/connected-applications."
        )
    return organizations


def get_org_dropdown(oauth_token: OAuthToken = None):
    if oauth_token is not None:
        orgs = list_orgs(oauth_token)
    else:
        orgs = []
    return gr.Dropdown(
        label="Organization",
        choices=orgs,
        value=orgs[0] if orgs else None,
        allow_custom_value=True,
        interactive=True,
    )


def get_token(oauth_token: OAuthToken = None):
    if oauth_token:
        return oauth_token.token
    else:
        return ""


def swap_visibility(oauth_token: Optional[OAuthToken] = None):
    if oauth_token:
        return gr.update(elem_classes=["main_ui_logged_in"])
    else:
        return gr.update(elem_classes=["main_ui_logged_out"])


def get_base_app():
    with gr.Blocks(
        title="🧬 Synthetic Data Generator",
        head="🧬  Synthetic Data Generator",
        css=_LOGGED_OUT_CSS,
    ) as app:
        with gr.Row():
            gr.Markdown(
                "Want to run this locally or with other LLMs? Take a look at the FAQ tab. distilabel Synthetic Data Generator is free, we use the authentication token to push the dataset to the Hugging Face Hub and not for data generation."
            )
        with gr.Row():
            gr.Column()
            get_login_button()
            gr.Column()

        gr.Markdown("## Iterate on a sample dataset")
        with gr.Column() as main_ui:
            pass

    return app


def get_argilla_client() -> Union[rg.Argilla, None]:
    try:
        api_url = os.getenv("ARGILLA_API_URL_SDG_REVIEWER")
        api_key = os.getenv("ARGILLA_API_KEY_SDG_REVIEWER")
        if api_url is None or api_key is None:
            api_url = os.getenv("ARGILLA_API_URL")
            api_key = os.getenv("ARGILLA_API_KEY")
        return rg.Argilla(
            api_url=api_url,
            api_key=api_key,
        )
    except Exception:
        return None

def get_preprocess_labels(labels: Optional[List[str]]) -> List[str]:
    return list(set([label.lower().strip() for label in labels])) if labels else []


def column_to_list(dataframe: pd.DataFrame, column_name: str) -> List[str]:
    if column_name in dataframe.columns:
        return dataframe[column_name].tolist()
    else:
        raise ValueError(f"Column '{column_name}' does not exist.")


def process_columns(
    dataframe,
    instruction_column: str,
    response_columns: Union[str, List[str]],
) -> List[dict]:
    instruction_column = [instruction_column]
    if isinstance(response_columns, str):
        response_columns = [response_columns]

    data = []
    for _, row in dataframe.iterrows():
        instruction = ""
        for col in instruction_column:
            value = row[col]
            if isinstance(value, (list, np.ndarray)):
                user_contents = [d["content"] for d in value if d.get("role") == "user"]
                if user_contents:
                    instruction = user_contents[-1]
            elif isinstance(value, str):
                try:
                    parsed_message = json.loads(value)
                    user_contents = [
                        d["content"] for d in parsed_message if d.get("role") == "user"
                    ]
                    if user_contents:
                        instruction = user_contents[-1]
                except json.JSONDecodeError:
                    instruction = value
            else:
                instruction = ""

        generations = []
        for col in response_columns:
            value = row[col]
            if isinstance(value, (list, np.ndarray)):
                if all(isinstance(item, dict) and "role" in item for item in value):
                    assistant_contents = [
                        d["content"] for d in value if d.get("role") == "assistant"
                    ]
                    if assistant_contents:
                        generations.append(assistant_contents[-1])
                else:
                    generations.extend(value)
            elif isinstance(value, str):
                try:
                    parsed_message = json.loads(value)
                    assistant_contents = [
                        d["content"]
                        for d in parsed_message
                        if d.get("role") == "assistant"
                    ]
                    if assistant_contents:
                        generations.append(assistant_contents[-1])
                except json.JSONDecodeError:
                    generations.append(value)
            else:
                pass

        data.append({"instruction": instruction, "generations": generations})

    return data


def extract_column_names(prompt_template: str) -> List[str]:
    env = Environment()
    parsed_content = env.parse(prompt_template)
    variables = meta.find_undeclared_variables(parsed_content)
    return list(variables)


def pad_or_truncate_list(lst, target_length):
    lst = lst or []
    lst_length = len(lst)
    if lst_length >= target_length:
        return lst[-target_length:]
    else:
        return lst + [None] * (target_length - lst_length)
