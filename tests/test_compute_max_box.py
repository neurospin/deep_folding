
from deep_folding.anatomist_tools.utils.bbox import compute_max

def test_compute_max_box():
    """Tests the function compte_max_box
    """
    # Upper right vertex of each box
    x1 = 10; y1 = -10; z1 = -100
    x2 = 0;  y2 = 100; z2 = 0
    list_bbmin = [[x1, y1, z1], [x2, y2, z2]]
    expected_bbmin = [0, -10, -100]

    # Lower left vertex of each box
    a1 = 20;  b1 = 0;   c1 = -50
    a2 = 10;  b2 = 110; c2 = 10
    list_bbmax = [[a1, b1, c1], [a2, b2, c2]]
    expected_bbmax = [20, 110, 10]

    # Performs the comparison
    bbmin, bbmax = compute_max(list_bbmin, list_bbmax)
    bbmin = bbmin.tolist()
    bbmax = bbmax.tolist()

    assert bbmin == expected_bbmin
    assert bbmax == expected_bbmax
