import numpy as np
import pandas as import pd

import pickle

from flask import Flask, request, render_template, jsonify

app = Flask('FNC')

@app.route('/')

def home():
    return "Welcome to my fake news classifier"

if __name__ == "__main__":
    app.run(debug=True)
