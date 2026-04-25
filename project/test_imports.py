import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize 
import implicit
from implicit.als import AlternatingLeastSquares
import joblib
import pickle
import os
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

print("Все библиотеки успешно импортированы.")
print(f"Python version: {os.sys.version}")
print(f"PyTorch version: {torch.__version__}") 
print(f"Implicit version: {implicit.__version__}") 