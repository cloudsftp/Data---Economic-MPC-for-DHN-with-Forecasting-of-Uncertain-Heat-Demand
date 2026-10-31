from mpc.config import CLUSTER_CONFIG, CONSUMER_DATA_DIRECOTRY
from mpc.demand import DemandProvider, DemandProviderI


def test_prediction_update_hours():
    base_hour = 15740
    num_points = 4

    def get_demand_provider(prediction_update_hours: int) -> DemandProviderI:
        demand_provider: DemandProviderI = DemandProvider(
            CONSUMER_DATA_DIRECOTRY,
            CLUSTER_CONFIG,
            base_hour,
            num_points,
            prediction_update_hours,
        )

        return demand_provider

    continuously_updating = get_demand_provider(1)
    sparingly_updating = get_demand_provider(4)

    continuously_updating_forecasts = []
    sparingly_updating_forecasts = []

    for i in range(10):
        continuously_updating_forecasts.append(
            continuously_updating.get_predicted_future_demands()["C1"]
        )
        sparingly_updating_forecasts.append(
            sparingly_updating.get_predicted_future_demands()["C1"]
        )

        continuously_updating.advance_step()
        sparingly_updating.advance_step()

    assert continuously_updating_forecasts[0] == sparingly_updating_forecasts[0]
    assert continuously_updating_forecasts[1] != sparingly_updating_forecasts[1]
    assert continuously_updating_forecasts[2] != sparingly_updating_forecasts[2]
    assert continuously_updating_forecasts[3] != sparingly_updating_forecasts[3]
    assert continuously_updating_forecasts[4] == sparingly_updating_forecasts[4]

    assert continuously_updating_forecasts[0][1] == sparingly_updating_forecasts[1][0]

    assert continuously_updating_forecasts[0][2] == sparingly_updating_forecasts[1][1]
    assert continuously_updating_forecasts[0][2] == sparingly_updating_forecasts[2][0]

    assert continuously_updating_forecasts[0][3] == sparingly_updating_forecasts[1][2]
    assert continuously_updating_forecasts[0][3] == sparingly_updating_forecasts[2][1]
    assert continuously_updating_forecasts[0][3] == sparingly_updating_forecasts[3][0]
