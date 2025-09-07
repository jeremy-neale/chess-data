Jeremy Neale Chess Dashboard

https://dashapp-1050976616574.us-east1.run.app/

Link live as of: August 7th, 2025

This dashboard is open-source and can be set up to run locally or deployed as a web application.

# Dataset

https://www.kaggle.com/datasets/arevel/chess-games

Download the dataset, unzip if needed, and place the CSV file in this directory.

# Usage

1. The requirements.txt needs to be installed - I would highly recommend a virtual environment using `venv`. Make sure to activate it, and it must be reactivated in the terminal each session that you want to run the python scripts.

2. To create the static plots, use `python3 static.py` or `python static.py`.

3. To create the dashboard locally:

- Make sure the constant at the top of app.py is set to 'LOCAL'.
- Run `python3 app.py` or `python app.py`

4. To deploy on a server like AWS or GCP, change run_type in app.py to 'SERVER'. The Dockerfile should be set up to run the application.

- Use Docker to build and then push. Then, deploy based on whatever cloud provider you choose.

# Notes

- The Dockerfile should not be needed to run app.py locally, it is for running on GCP or similar. I am including it in case someone wants to deploy it.
- I'm including the entire dataset, but only load in the first 50k entries by default. This can be changed in the code.

# Main Conclusions from first 50,000 games

• Draws tend to have more checks.

• Higher elo players tend to castle earlier.

• White tends to castle and move its queen first.

• Black tends to make the first capture.

• Turn 7 is the most common turn for the first castle to occur (minimum moves to castle is 4).

• There is a minor positive correlation (r=0.26) between the first turn a capture occurs and the first turn a
queen is moved.

• Black tends to castle queenside more frequently than white.

• Kingside castle is more common for both black and white.

• White wins more games than black.

• Total captures is fairly evenly distributed between 0-30 (no more or less can occur outside this range).

• Total checks is skewed right.

• There is a negative correlation between the first turn a capture occurs and the first turn a castle occurs.

• Players in longer time formats (Classical) tend to bring out their queen earlier than shorter time formats
(Blitz, Bullet).

• Draws are more common for players with higher elos.

• Certain openings are better for certain colors at different elo ranges.

• Certain ECO openings are more prone to draws, and this range tends to be at higher elos.


These conclusions are observational, so they are NOT conclusive cause-effect relationships by any mean.
