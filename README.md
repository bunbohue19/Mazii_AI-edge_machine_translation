
# Project Structure

```
.
|-- README.md
|-- requirements.txt
`-- src
    |-- inference
    |   |-- app.py
    |   |-- app.sh
    |   |-- base_model
    |   |   |-- __init__.py
    |   |   |-- chat.py
    |   |   |-- completion.py
    |   |   `-- translation.py
    |   |-- client.py
    |   |-- config
    |   |   `-- config.yml
    |   |-- main.py
    |   |-- main.sh
    |   `-- server.py
    |-- synthesize-data
    |   |-- combine.py
    |   `-- ja2xx
    |       |-- translation_runner.py
    |       `-- translation_runner.sh
    |-- test
    |   `-- test.sh
    `-- train
        `-- SFT
            |-- main.py
            |-- main.sh
            `-- unsloth_compiled_cache
                |-- UnslothBCOTrainer.py
                |-- UnslothCPOTrainer.py
                |-- UnslothDPOTrainer.py
                |-- UnslothGKDTrainer.py
                |-- UnslothGRPOTrainer.py
                |-- UnslothKTOTrainer.py
                |-- UnslothNashMDTrainer.py
                |-- UnslothOnlineDPOTrainer.py
                |-- UnslothORPOTrainer.py
                |-- UnslothPPOTrainer.py
                |-- UnslothPRMTrainer.py
                |-- UnslothRLOOTrainer.py
                |-- UnslothRewardTrainer.py
                |-- UnslothSFTTrainer.py
                `-- UnslothXPOTrainer.py
```
