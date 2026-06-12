# Converter/validator environment for re-exporting old Prosit models to CPU-capable SavedModels.
# Host Python can't run TF; build and run conversions inside this image.
#   docker build -t koina-convert -f tools/cpu_reexport/Dockerfile tools/cpu_reexport
#   docker run --rm -v "$PWD:/w" koina-convert \
#     python /w/tools/cpu_reexport/reexport_prosit_intensity.py \
#       /w/models/Prosit/Prosit_2019_intensity_core/1/model.savedmodel /w/out/cpu_savedmodel
FROM python:3.11-slim
RUN pip install --no-cache-dir 'numpy<2' tensorflow-cpu==2.15.1
