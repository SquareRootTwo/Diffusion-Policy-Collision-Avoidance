FROM curobo_docker:isaac_sim_2023.1.0

# install python packages
RUN /isaac-sim/python.sh -m pip install --upgrade pip
RUN /isaac-sim/python.sh -m pip install fastparquet
RUN /isaac-sim/python.sh -m pip install pyarrow
RUN /isaac-sim/python.sh -m pip install pandas

# run the python dataset generation script
ENTRYPOINT ["/isaac-sim/python.sh", "/src/main.py"]