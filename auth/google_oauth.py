# auth/google_oauth.py

import os
import streamlit as st
import requests
from urllib.parse import urlencode

def get_login_url():
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    redirect_uri = "https://medibot-kvlmhodraxtdebnqqqhxum.streamlit.app"
    scope = "openid email profile"
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth"

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "response_type": "code",
        "access_type": "offline",
        "prompt": "consent"
    }

    return f"{auth_url}?{urlencode(params)}"

def fetch_tokens(code):
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = "http://localhost:8501"

    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"
    }

    response = requests.post(token_url, data=data)
    return response.json()

def get_user_info(access_token):
    response = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    return response.json()
