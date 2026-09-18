# Failure analysis - Component 1 (test set)

- Images evaluated: 431
- Disc not detected: 0
- Cup not detected: 0
- Dice without post-processing: disc 0.9502, cup 0.8354
- Dice with post-processing: disc 0.9513, cup 0.8357

## Dice by source dataset

```
                  dice_disc       dice_cup      
                       mean count     mean count
source                                          
CRFO-v4              0.9669     7   0.7447     7
DRISHTI-GS1-test     0.9656     8   0.8790     8
DRISHTI-GS1-train    0.9727     7   0.8151     7
G1020                0.9489   119   0.8198   119
ORIGA                0.9550    97   0.8779    97
PAPILA               0.9478    73   0.7280    73
REFUGE1-train        0.9538    60   0.8828    60
REFUGE1-val          0.9456    60   0.8903    60
```

## 10 worst cup segmentations

```
      name  dice_disc  dice_cup  pred_vcdr  gt_vcdr
 PAPILA-50     0.9589    0.0690     0.1667   0.1188
 G1020-667     0.8545    0.1762     0.1481   0.3478
PAPILA-301     0.9545    0.1945     0.2430   0.1143
PAPILA-341     0.9461    0.2216     0.2333   0.1124
PAPILA-128     0.9625    0.3091     0.3371   0.1724
CRFO-v4-48     0.9739    0.3138     0.4583   0.2740
PAPILA-165     0.9454    0.3353     0.2872   0.1573
 PAPILA-47     0.9779    0.3784     0.1489   0.2872
PAPILA-441     0.9563    0.3787     0.2093   0.0920
PAPILA-349     0.9372    0.4115     0.2198   0.1364
```

## Large CDR errors (|error| > 0.15): 21 images

```
                name  dice_disc  dice_cup  pred_vcdr  gt_vcdr  abs_vcdr_error
           ORIGA-333     0.8891    0.7034     0.3671   0.6757          0.3086
 DRISHTI-GS1-train-7     0.9798    0.6571     0.6092   0.9091          0.2999
           ORIGA-387     0.9371    0.8049     0.5500   0.8158          0.2658
           ORIGA-327     0.8944    0.7708     0.6351   0.8824          0.2473
           G1020-552     0.9412    0.5150     0.3976   0.6437          0.2461
           ORIGA-511     0.8611    0.8885     0.6795   0.9000          0.2205
DRISHTI-GS1-train-40     0.9870    0.7753     0.6765   0.8932          0.2167
           ORIGA-559     0.9826    0.6999     0.4430   0.6538          0.2108
           G1020-667     0.8545    0.1762     0.1481   0.3478          0.1997
           ORIGA-560     0.9858    0.6444     0.4474   0.6456          0.1982
           ORIGA-616     0.9574    0.6746     0.4744   0.2763          0.1981
             ORIGA-4     0.9682    0.7790     0.5263   0.7200          0.1937
           G1020-327     0.9592    0.6642     0.5802   0.3947          0.1855
          CRFO-v4-48     0.9739    0.3138     0.4583   0.2740          0.1843
           G1020-504     0.9614    0.5858     0.4459   0.6301          0.1842
```