from drivers.models.main_models import main_models
from drivers.plots.main_plots import main_plots
from drivers.test_statistics import main_test_statistics
from drivers.render_model_tables import render_all 


def main():
    main_models()
    main_plots()
    main_test_statistics()
    render_all() 

if __name__ == "__main__":
    main()