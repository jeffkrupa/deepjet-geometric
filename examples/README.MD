To run fully supervised:

```. train.sh --nepochs 200 --ntrain 1e6 --which_augmentations 0 1 2 3 --fully_supervised```

To run fine-tuning:

```. train.sh --nepochs 200 --ntrain 1e6 --which_augmentations 0 1 2 3 --mpath ./mar20_run0/best-epoch.pt``` (must supply mpath in this case)

other args:
`--one_layer_MLP`
`--fix_weights`
