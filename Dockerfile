FROM rayproject/ray-ml:1.6.0

# Copy watts/* into into /home/ray/
COPY . .

RUN pip install -e .
