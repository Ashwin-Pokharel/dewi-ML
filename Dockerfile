FROM tensorflow/serving
COPY models/ models/saved_model
ENV MODEL_NAME=saved_model
EXPOSE 8500
EXPOSE 8501