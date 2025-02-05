import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from src.exception import CustomException
import traceback
import logging
import datetime

class RunVisPipeline:
    def __init__(self, model_path=os.path.join('artifacts', 'model.h5')):
        self.model_path = model_path

    def predict(self, MF=1.0):
        try:
            model = tf.keras.models.load_model(self.model_path)
            EX = pd.read_csv(os.path.join('artifacts', 'test.csv'))
            X_test = EX.drop(['master_signal'], axis='columns')
            y_pred_series = pd.read_csv(os.path.join('artifacts', 'predicted_probablity.csv'))
            #y_pred_mean = y_pred_series.mean()
            y_pred_new = y_pred_series
            y_pred_new.columns = ['Prediction']  # Rename prediction column

            X_test_new = X_test.reset_index(drop=True)
            y_pred_new_series = y_pred_new.reset_index(drop=True)

            # Logging predictions and features
            logging.info(f'X_test_new: {X_test_new}')
            logging.info(f'y_pred_new_series: {y_pred_new_series}')
            #logging.info(f'y_pred_mean: {y_pred_mean}')

            return X_test_new, y_pred_new_series
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def concatenate_predictions_with_features(X_test_new, y_pred_new_series):
        concatenated_df = pd.concat([X_test_new, y_pred_new_series], axis=1)
        return concatenated_df

    @staticmethod
    def preprocess_and_visualize(concatenated_df, x=None, y=None):
        try:
            date_time_df = pd.DataFrame({
                'Year': concatenated_df['Year'],
                'Month': concatenated_df['Month'],
                'Day': concatenated_df['Day'],
                'Hour': concatenated_df['Hour'],
                'Minute': concatenated_df['Minute']
            })

            date_time_df['Datetime'] = pd.to_datetime(date_time_df)
            date_time_df['Datetime'] = date_time_df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            date_time_df['Datetime'] = date_time_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S%z')

            # Create a datetime index
            date_time_df['Datetime'] = pd.to_datetime(date_time_df['Datetime'])
            concatenated_df['Datetime'] = date_time_df['Datetime']

            columns_to_drop = ['Year', 'Month', 'Day', 'Hour', 'Minute']
            concatenated_df = concatenated_df.drop(columns=columns_to_drop)

            concatenated_df.set_index('Datetime', inplace=True)

            concatenated_df.to_csv('check2.csv')

            data = pd.read_csv(os.path.join('artifacts', 'df_new.csv'))
            logging.info(f'concatdf: {concatenated_df}')

            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data.set_index('Datetime', inplace=True)

            concatenated_df_new = data.merge(concatenated_df, left_index=True, right_index=True, how='left')

            concatenated_df_new = concatenated_df_new.drop(columns=['EMASignal', 'isPivot', 'CHOCH_pattern_detected',
                                  'fibonacci_signal','MinSwing','MaxSwing','SL','TP',
                                 'LBD_detected','LBH_detected','SR_signal','isBreakOut','candlestick_signal',
                                 'result','signal1','buy_signal','Position','sell_signal','fractal_high',
                                 'fractals_low','VSignal','PriceSignal','TotSignal','SLSignal','grid_signal',
                                 'ordersignal','SLSignal_heiken','EMASignal1','long_signal','martiangle_signal',
                                                     'fractal_low','buy_signal1','sell_signal1','fractals_high','Candle_direction'])
            
            concatenated_df_new = concatenated_df_new.fillna(0)
            concatenated_df_new['Prediction'] = concatenated_df_new['Prediction'].astype(int)

            def pointpos(x):
                if x['Prediction'] == 1:
                    return (x['Low'] + 0.5e-3)
                elif x['Prediction'] == 2: 
                    return (x['Low'] + 0.5e-3)
                elif x['Prediction'] == 3: 
                    return (x['Low'] + 0.5e-3)
                elif x['Prediction'] == 4: 
                    return (x['Low'] + 0.5e-3)
                elif x['Prediction'] == 5: 
                    return (x['Low'] + 0.5e-3)
                else:
                    return np.nan

            concatenated_df_new['pointpos'] = concatenated_df_new.apply(lambda row: pointpos(row), axis=1)

            concatenated_df_new.to_csv('check1.csv')

            logging.info(f'concatdfn: {concatenated_df_new}')
            logging.info(concatenated_df_new[concatenated_df_new['Prediction'] != 0])

            # Calculate moving averages
            concatenated_df_new['SMA_50'] = concatenated_df_new['Close'].rolling(window=50).mean()
            concatenated_df_new['SMA_200'] = concatenated_df_new['Close'].rolling(window=200).mean()

            # Filter based on datetime range
            concatenated_df_new_filtered = concatenated_df_new.loc[x:y]

            sell_signals = concatenated_df_new_filtered[concatenated_df_new_filtered['Prediction'] == 1]
            buy_signals = concatenated_df_new_filtered[concatenated_df_new_filtered['Prediction'] == 2]
            buy_signals3 = concatenated_df_new_filtered[concatenated_df_new_filtered['Prediction'] == 3]
            buy_signals4 = concatenated_df_new_filtered[concatenated_df_new_filtered['Prediction'] == 4]
            buy_signals5 = concatenated_df_new_filtered[concatenated_df_new_filtered['Prediction'] == 5]

            fig = go.Figure(data=[go.Candlestick(
                x=concatenated_df_new_filtered.index,
                open=concatenated_df_new_filtered['Open'],
                high=concatenated_df_new_filtered['High'],
                low=concatenated_df_new_filtered['Low'],
                close=concatenated_df_new_filtered['Close']
            )])

            # Add moving averages
            fig.add_trace(go.Scatter(
                x=concatenated_df_new_filtered.index,
                y=concatenated_df_new_filtered['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=concatenated_df_new_filtered.index,
                y=concatenated_df_new_filtered['SMA_200'],
                mode='lines',
                name='SMA 200',
                line=dict(color='blue', width=2)
            ))

            # Add prediction points

# Plot Sell Signal
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['pointpos'],
                mode="markers",
                marker=dict(
                    size=8,
                    color="cyan",  # Gold color for sell signals
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name="Sell Signal"
            ))

# Plot Buy Signal
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['pointpos'],
                mode="markers",
                marker=dict(
                    size=8,
                    color="MediumPurple",  # Cyan color for buy signals
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name="Buy Signal"
            ))

            fig.add_trace(go.Scatter(
                x=buy_signals3.index,
                y=buy_signals3['pointpos'],
                mode="markers",
                marker=dict(
                    size=8,
                    color="orange",  # Cyan color for buy signals
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name="Buy Signal3"
            ))

            fig.add_trace(go.Scatter(
                x=buy_signals4.index,
                y=buy_signals4['pointpos'],
                mode="markers",
                marker=dict(
                    size=8,
                    color="yellow",  # Cyan color for buy signals
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name="Buy Signal4"
            ))

            fig.add_trace(go.Scatter(
                x=buy_signals5.index,
                y=buy_signals5['pointpos'],
                mode="markers",
                marker=dict(
                    size=8,
                    color="Green",  # Cyan color for buy signals
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name="Buy Signal5"
            ))


            # Add volume bars
            fig.add_trace(go.Bar(
                x=concatenated_df_new_filtered.index,
                y=concatenated_df_new_filtered['Volume'],
                name='Volume',
                marker_color='gray',
                opacity=0.3,
                yaxis='y2'
            ))

            filtered_lows = concatenated_df_new_filtered[concatenated_df_new_filtered['Low'] != 0]['Low']
            y_min = filtered_lows.min()
            y_max = concatenated_df_new_filtered['High'].max()

            fig.update_layout(
                title="MASTER'S STRATEGY",
                xaxis_title="Date",
                yaxis_title="Price",
                autosize=False,
                width=950,
                height=600,
                margin=dict(l=50, r=50, b=100, t=100, pad=4),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgb(30,30,30)',
                font=dict(color="white"),
                xaxis=dict(
                    gridcolor='rgb(50,50,50)',
                    rangeslider=dict(visible=True),  # Add range slider
                    rangeselector=dict(              # Add range selector
                        buttons=list([
                            dict(count=1, label='5m', step='minute', stepmode='backward'),
                            dict(count=5, label='15m', step='minute', stepmode='backward'),
                            dict(count=15, label='45m', step='minute', stepmode='backward'),
                            dict(count=1, label='1h', step='hour', stepmode='backward'),
                            dict(count=1, label='1d', step='day', stepmode='backward'),
                            dict(count=30, label='30d', step='day', stepmode='backward'),
                            dict(step='all')
                            ]),
                            bgcolor='black',  # Change the background color of the buttons to black
                            activecolor='darkgrey',  # Change the background color of the active button
                            font=dict(color='white')  # Change the font color of the buttons to white)
                    )
                ),
                yaxis=dict(
                    gridcolor='rgb(50,50,50)',
                    range=[y_min, y_max]
                ),
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Align the legend vertically
                 y=1.02,  # Position the legend just above the plot
                 xanchor="center",  # Align the legend horizontally
                 x=0.8  # Center the legend
                 )
            )

            fig.update_xaxes(
                rangeslider_visible=True, 
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label='5m', step='minute', stepmode='backward'),
                        dict(count=5, label='15m', step='minute', stepmode='backward'),
                        dict(count=15, label='45m', step='minute', stepmode='backward'),
                        dict(count=1, label='1h', step='hour', stepmode='backward'),
                        dict(count=1, label='1d', step='day', stepmode='backward'),
                        dict(count=30, label='30d', step='day', stepmode='backward'),
                        dict(step='all')
                    ])
                )
            )

            fig.write_html(os.path.join('templates', 'plot.html'))

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self, MF=1.0, x=None, y=None):
        try:
            X_test_new, y_pred_new_series = self.predict(MF=MF)
            concatenated_df = self.concatenate_predictions_with_features(X_test_new, y_pred_new_series)
            self.preprocess_and_visualize(concatenated_df, x=x, y=y)
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    train_pipeline = RunVisPipeline()
    train_pipeline.run_pipeline()
