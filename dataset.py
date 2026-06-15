{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP8Ghn4aNvlTQ9jA5SGfewI",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/andriakobaidze/ml-assn-04/blob/main/dataset.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Xi-uq0JrvW7P",
        "outputId": "1b601ed0-7392-4406-c631-3fcc364179fe"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/notebook/notebookapp.py:191: SyntaxWarning: invalid escape sequence '\\/'\n",
            "  | |_| | '_ \\/ _` / _` |  _/ -_)\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: (1) Create a W&B account\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: (2) Use an existing W&B account\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: (3) Don't visualize my results\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[34m\u001b[1mwandb\u001b[0m: Enter your choice: 2\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mwandb\u001b[0m: You chose 'Use an existing W&B account'\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: Logging into https://api.wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: Create a new API key at: https://wandb.ai/authorize?ref=models\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: Store your API key securely and do not share it.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[34m\u001b[1mwandb\u001b[0m: Paste your API key and hit enter: ··········\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mwandb\u001b[0m: No netrc file found, creating one.\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: Appending key for api.wandb.ai to your netrc file: /root/.netrc\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: Currently logged in as: \u001b[33makoba23\u001b[0m (\u001b[33makoba23-free-university-of-tbilisi-\u001b[0m) to \u001b[32mhttps://api.wandb.ai\u001b[0m. Use \u001b[1m`wandb login --relogin`\u001b[0m to force relogin\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 1
        }
      ],
      "source": [
        "!pip install wandb -q\n",
        "\n",
        "import wandb\n",
        "wandb.login()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "run = wandb.init(project=\"ML-assn-02\", name=\"test-run\")\n",
        "wandb.log({\"test_metric\": 1.0})\n",
        "run.finish()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 510
        },
        "id": "Au329Vs21lAu",
        "outputId": "1fd72012-842e-4c5a-baec-8948108fc0a8"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": []
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "Tracking run with wandb version 0.27.2"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "Run data is saved locally in <code>/content/wandb/run-20260615_195527-egl959ky</code>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "Syncing run <strong><a href='https://wandb.ai/akoba23-free-university-of-tbilisi-/ML-assn-02/runs/egl959ky' target=\"_blank\">test-run</a></strong> to <a href='https://wandb.ai/akoba23-free-university-of-tbilisi-/ML-assn-02' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target=\"_blank\">docs</a>)<br>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              " View project at <a href='https://wandb.ai/akoba23-free-university-of-tbilisi-/ML-assn-02' target=\"_blank\">https://wandb.ai/akoba23-free-university-of-tbilisi-/ML-assn-02</a>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              " View run at <a href='https://wandb.ai/akoba23-free-university-of-tbilisi-/ML-assn-02/runs/egl959ky' target=\"_blank\">https://wandb.ai/akoba23-free-university-of-tbilisi-/ML-assn-02/runs/egl959ky</a>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": []
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>test_metric</td><td>▁</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>test_metric</td><td>1</td></tr></table><br/></div></div>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              " View run <strong style=\"color:#cdcd00\">test-run</strong> at: <a href='https://wandb.ai/akoba23-free-university-of-tbilisi-/ML-assn-02/runs/egl959ky' target=\"_blank\">https://wandb.ai/akoba23-free-university-of-tbilisi-/ML-assn-02/runs/egl959ky</a><br> View project at: <a href='https://wandb.ai/akoba23-free-university-of-tbilisi-/ML-assn-02' target=\"_blank\">https://wandb.ai/akoba23-free-university-of-tbilisi-/ML-assn-02</a><br>Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "Find logs at: <code>./wandb/run-20260615_195527-egl959ky/logs</code>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XFO26SD97Fej",
        "outputId": "1ccae5cf-3806-40be-f2ff-e990edbfa0cd"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls /content/drive/MyDrive/ml-assn-04/\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DnF9d6qU8Y5k",
        "outputId": "3db5c53d-4870-47e2-b3d7-b53268f6f1ef"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "example_submission.csv\tfer2013.tar.gz\ticml_face_data.csv  test.csv  train.csv\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "df = pd.read_csv('/content/drive/MyDrive/ml-assn-04/train.csv')\n",
        "print(df.head())\n",
        "print(df.shape)\n",
        "print(df.columns.tolist())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6msk-Vru91kH",
        "outputId": "65768c71-5ef4-4233-a8a5-ad4e762e2320"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   emotion                                             pixels\n",
            "0        0  70 80 82 72 58 58 60 63 54 58 60 48 89 115 121...\n",
            "1        0  151 150 147 155 148 133 111 140 170 174 182 15...\n",
            "2        2  231 212 156 164 174 138 161 173 182 200 106 38...\n",
            "3        4  24 32 36 30 32 23 19 20 30 41 21 22 32 34 21 1...\n",
            "4        6  4 0 0 0 0 0 0 0 0 0 0 0 3 15 23 28 48 50 58 84...\n",
            "(28709, 2)\n",
            "['emotion', 'pixels']\n"
          ]
        }
      ]
    }
  ]
}