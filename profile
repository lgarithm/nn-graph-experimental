--------------------------------------------------------------------------------
    #     count      cumulative (s)               %       mean (ms)    call site
--------------------------------------------------------------------------------
    1         1           22.348874          100.00      22348.8740    ./bin/train-slp
    2        20           22.024634           98.55       1101.2317    train epoch
    3         1           21.405842           95.78      21405.8417    slp_example
    4       120           18.028282           80.67        150.2357    train
    5        60           13.792600           61.71        229.8767    nn::experimental::ops::grad::matmul<1, nn::engines::plain>
    6       120            6.947723           31.09         57.8977    nn::ops::matmul_<nn::engines::plain>
    7       120            3.996297           17.88         33.3025    test
    8         1            0.943029            4.22        943.0289    slp_gpu
    9       120            0.416735            1.86          3.4728    nn::ops::softmax
   10        60            0.300570            1.34          5.0095    nn::cuda::ops::similarity
   11        60            0.086394            0.39          1.4399    nn::experimental::ops::grad::softmax<0>
   12        60            0.025607            0.11          0.4268    nn::ops::xentropy
   13       120            0.009525            0.04          0.0794    nn::experimental::ops::argmax
   14       120            0.006332            0.03          0.0528    nn::ops::apply_bias<nn::ops::hw, nn::ops::scalar_add>
   15        60            0.005551            0.02          0.0925    nn::experimental::ops::grad::xentropy<1>
   16        60            0.002073            0.01          0.0345    nn::experimental::ops::grad::add_bias<nn::ops::hw, 1>
   17        60            0.000985            0.00          0.0164    nn::experimental::ops::grad::add_bias<nn::ops::hw, 0>
   18       120            0.000443            0.00          0.0037    nn::cuda::ops::mm
   19        60            0.000418            0.00          0.0070    nn::cuda::ops::grad::add_bias<nn::traits::hw, 0>
   20       120            0.000317            0.00          0.0026    nn::cuda::ops::add_bias<nn::traits::hw>
   21       120            0.000253            0.00          0.0021    nn::cuda::ops::softmax
   22       120            0.000246            0.00          0.0021    nn::cuda::ops::argmax
   23        60            0.000204            0.00          0.0034    nn::cuda::ops::grad::softmax<0>
   24        60            0.000174            0.00          0.0029    nn::cuda::ops::grad::add_bias<nn::traits::hw, 1>
   25        60            0.000170            0.00          0.0028    nn::cuda::ops::grad::xentropy<1>
   26        60            0.000167            0.00          0.0028    nn::cuda::ops::grad::matmul<1>
   27        60            0.000143            0.00          0.0024    nn::cuda::ops::xentropy
   28        60            0.000124            0.00          0.0021    nn::experimental::ops::similarity
   29         2            0.000050            0.00          0.0248    slp
