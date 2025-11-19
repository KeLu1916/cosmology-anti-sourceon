
import numpy as np
from src.antiosourceon.core import AntiSourceonModel

class TestAntiSourceonModel:
    """Anti-sourceon model unit tests"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = AntiSourceonModel(DeltaE_eV=2.07, g_int=6.95e-5)
        
        assert model.DeltaE_eV == 2.07
        assert model.g_int == 6.95e-5
        assert model.f_ede_max == 0.1
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test invalid energy barrier
        with pytest.raises(ValueError):
            AntiSourceonModel(DeltaE_eV=0.001)
        
        # Test invalid coupling constant
        with pytest.raises(ValueError):
            AntiSourceonModel(g_int=1e-12)
    
    def test_decoupling_calculation(self):
        """Test decoupling parameter calculation"""
        model = AntiSourceonModel(DeltaE_eV=2.07)
        
        assert 3000 <= model.z_dec <= 10000
        assert 8000 <= model.T_dec <= 30000
        assert isinstance(model.z_dec, float)
        assert isinstance(model.T_dec, float)
    
    def test_ede_density_fraction(self):
        """Test EDE density fraction calculation"""
        model = AntiSourceonModel()
        
        # High redshift should have higher EDE fraction
        f_high_z = model.ede_density_fraction(10000)
        f_low_z = model.ede_density_fraction(100)
        
        assert 0 <= f_high_z <= 0.2
        assert 0 <= f_low_z <= 0.2
        assert f_high_z > f_low_z
    
    def test_equation_of_state(self):
        """Test equation of state parameter"""
        model = AntiSourceonModel()
        
        w_high_z = model.ede_equation_of_state(10000)
        w_low_z = model.ede_equation_of_state(100)
        
        assert -1.0 <= w_high_z <= 0
        assert -1.0 <= w_low_z <= 0
    
    def test_interaction_rate(self):
        """Test interaction rate"""
        model = AntiSourceonModel()
        
        # Test reaction rates at different temperatures
        Gamma_high = model.interaction_rate(20000)  # High temperature
        Gamma_low = model.interaction_rate(1000)    # Low temperature
        
        assert Gamma_high > 0
        assert Gamma_low > 0
        assert Gamma_high > Gamma_low
    
    def test_physical_parameters(self):
        """Test physical parameter retrieval"""
        model = AntiSourceonModel()
        params = model.get_physical_parameters()
        
        expected_keys = ['DeltaE_eV', 'g_int', 'f_ede_max', 'z_dec', 'T_dec', 'DeltaE_J']
        for key in expected_keys:
            assert key in params
            assert isinstance(params[key], (int, float))