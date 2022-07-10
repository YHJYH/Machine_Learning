## current observations
- max pooling layer is important
    - exp: deduce which layers contain max_pool in raw VGG16 can be deducted, see what happens if reduce layers with max_pool (control number of layers or params equivalent)
- fixed initialization: try not fixed initialization. 
- based on ablation study, shall draw acc vs. % of parameters reduction and acc vs. num of layers.

Based on the plot:<br>
![acc_vs_percent_removal](acc_percent_param.png) and <br>
![u_shape](U.png)
