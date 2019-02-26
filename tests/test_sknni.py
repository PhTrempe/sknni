from unittest import TestCase

from sknni import SkNNI, MAX_LAT, MIN_LAT, MIN_LNG, MAX_LNG


class TestSknni(TestCase):
    def setUp(self):
        self.sknni = SkNNI([(0, 0, 0)])
        self.coords = [
            (-90, 0), (-90, 90), (-90, -180), (-90, -90),
            (-45, 0), (-45, 90), (-45, -180), (-45, -90),
            (0, 0), (0, 90), (0, -180), (0, -90),
            (45, 0), (45, 90), (45, -180), (45, -90),
            (MAX_LAT, 0), (MAX_LAT, 90), (MAX_LAT, -180), (MAX_LAT, -90)
        ]

    def tearDown(self):
        del self.sknni
        del self.coords

    def test_sknni_init_obs_none(self):
        with self.assertRaises(ValueError):
            SkNNI(None)

    def test_sknni_init_obs_int(self):
        with self.assertRaises(ValueError):
            SkNNI(42)

    def test_sknni_init_obs_str(self):
        with self.assertRaises(ValueError):
            SkNNI('')

    def test_sknni_init_obs_empty_list(self):
        with self.assertRaises(ValueError):
            SkNNI([])

    def test_sknni_init_obs_ok(self):
        SkNNI([(0, 0, 0)])

    def test_sknni_init_obs_not_tuple(self):
        with self.assertRaises(ValueError):
            SkNNI([(0, 0, 0), 42])

    def test_sknni_init_obs_list_empty_obs(self):
        with self.assertRaises(ValueError):
            SkNNI([(0, 0, 0), ()])

    def test_sknni_init_obs_list_obs_wrong_len(self):
        with self.assertRaises(ValueError):
            SkNNI([(0, 0, 0), (0, 0)])

    def test_sknni_init_r_wrong_type(self):
        with self.assertRaises(ValueError):
            SkNNI([(0, 0, 0)], r='42')

    def test_sknni_init_r_0(self):
        with self.assertRaises(ValueError):
            SkNNI([(0, 0, 0)], r=0)

    def test_sknni_init_r_negative(self):
        with self.assertRaises(ValueError):
            SkNNI([(0, 0, 0)], r=-1)

    def test_sknni_init_r_positive(self):
        SkNNI([(0, 0, 0)], r=1)

    def test_sknni_interp_coords_list_empty(self):
        with self.assertRaises(ValueError):
            self.sknni([])

    def test_sknni_interp_coords_ok(self):
        self.sknni([(0, 0)])

    def test_sknni_interp_coords_coord_empty_tuple(self):
        with self.assertRaises(ValueError):
            self.sknni([(0, 0), ()])

    def test_sknni_interp_coords_coord_wrong_len(self):
        with self.assertRaises(ValueError):
            self.sknni([(0, 0), (0,)])

    def test_sknni_interp_k_wrong_type(self):
        with self.assertRaises(ValueError):
            self.sknni([(0, 0)], k='42')

    def test_sknni_interp_k_zero(self):
        with self.assertRaises(ValueError):
            self.sknni([(0, 0)], k=0)

    def test_sknni_interp_k_negative(self):
        with self.assertRaises(ValueError):
            self.sknni([(0, 0)], k=-1)

    def test_sknni_interp_k_positive(self):
        self.sknni([(0, 0)], k=1)

    def test_sknni_interp_interp_fn(self):
        with self.assertRaises(ValueError):
            self.sknni([(0, 0)], interp_fn='42')

    def test_sknni_one_point(self):
        for (la, lo) in self.coords:
            for (la2, lo2) in self.coords:
                for v in (-5, 0, 5):
                    interp_v = SkNNI([(la, lo, v)])([(la2, lo2)])[0][2]
                    self.assertAlmostEqual(interp_v, v)

    def test_sknni_extreme_lngs(self):
        for v in [-10, -5, 0, 5, 10]:
            obs = [(MAX_LAT, 0, v), (0, -90, v), (MIN_LAT, 0, v), (0, 90, v)]
            sknni = SkNNI(obs)
            interps = sknni([(0, MIN_LNG), (0, MAX_LNG)])
            for _, _, interp_val in interps:
                self.assertAlmostEqual(interp_val, v)

    def test_sknni_two_obs_at_same_point(self):
        for (la, lo) in self.coords:
            for v in (-5, 0, 5):
                obs = [(la, lo, v), (la, lo, 2 * v)]
                interp_v = SkNNI(obs)([(la, lo)])[0][2]
                self.assertAlmostEqual(interp_v, 3 * v / 2)

    def test_sknni_obs_pole_interp_equator(self):
        for v1 in (-5, 0, 5):
            for v2 in (-5, 0, 5):
                obs = [(-90, 0, v1), (MAX_LAT, 0, v2)]
                sknni = SkNNI(obs)
                for (la, lo) in [(0, lng) for lng in (-180, -90, 0, 90)]:
                    interp_v = sknni([(la, lo)])[0][2]
                    self.assertAlmostEqual(interp_v, (v1 + v2) / 2)

    def test_sknni_obs_equator_interp_south_pole(self):
        obs = [(0, -180, 0), (0, -90, 0), (0, 0, 0), (0, 90, 0)]
        sknni = SkNNI(obs)
        v = sknni([(MIN_LAT, 0)])[0][2]
        for lng in range(-180, 180, 30):
            interp_v = sknni([(-90, lng)])[0][2]
            self.assertAlmostEqual(interp_v, v)

    def test_sknni_obs_equator_interp_north_pole(self):
        obs = [(0, -180, 0), (0, -90, 0), (0, 0, 0), (0, 90, 0)]
        sknni = SkNNI(obs)
        v = sknni([(MAX_LAT, 0)])[0][2]
        for lng in range(-180, 180, 30):
            interp_v = sknni([(-90, lng)])[0][2]
            self.assertAlmostEqual(interp_v, v)
