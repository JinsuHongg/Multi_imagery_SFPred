# Introduction

Command Example,   
python -m Multi_imagery_SFPred.Optimization.Solarflare_4cls_CV_search --models Mobilenet --data All  

In this project, we implement a binary classification task using multi-imagery datasets. Existing research has primarily focused on single-channel images for full-disk solar flare prediction. Interestingly, these studies have shown that simpler models, such as AlexNet, often outperform more complex ones. However, there may be untapped information in the solar surface or atmosphere that single-channel images fail to capture. We hypothesize that utilizing multi-channel imagery can address data insufficiency issues in deep learning, allowing more complex models to achieve higher performance.