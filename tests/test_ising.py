import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import openpyxl

from smtj_ising.model import Ising, maxcut, cut_weight, qubo, exact, ferromagnetic_minimum
from smtj_ising.measurements import Record, causal_residual, prepare_replay, load_records
from smtj_ising.solver import solve, MODES


class PhysicsAndEncodingTests(unittest.TestCase):
    def test_delta_matches_full_energy_with_fields(self):
        rng=np.random.default_rng(8)
        a=rng.normal(size=(7,7));a=(a+a.T)/2;np.fill_diagonal(a,0)
        p=Ising(a,rng.normal(size=7))
        for state in itertools.product([-1,1],repeat=7):
            s=np.array(state)
            for i in range(7):
                t=s.copy();t[i]*=-1
                self.assertAlmostEqual(p.delta(s,i),p.energy(t)-p.energy(s))

    def test_maxcut_sign_and_offset(self):
        w=np.array([[0,2,3],[2,0,4],[3,4,0]])
        p=maxcut(w)
        e,s=exact(p)
        self.assertEqual(cut_weight(p,s),7)
        for state in itertools.product([-1,1],repeat=3):
            self.assertEqual(p.energy(state),np.triu(w,1).sum()-2*cut_weight(p,state))

    def test_qubo_full_and_upper_triangular_conventions(self):
        for Q in (np.array([[3,2,1],[0,-2,5],[0,0,4.]]), np.array([[1,2,3],[2,4,-1],[3,-1,5.]])):
            p,offset=qubo(Q)
            for state in itertools.product([-1,1],repeat=3):
                s=np.array(state);x=(s+1)/2
                self.assertAlmostEqual(x@Q@x,p.energy(s)+offset)

    def test_graph_cut_against_enumeration(self):
        for seed in range(8):
            rng=np.random.default_rng(seed)
            a=np.triu(rng.integers(0,4,(8,8)),1);p=Ising(a+a.T,rng.normal(size=8))
            e,s=ferromagnetic_minimum(p)
            self.assertAlmostEqual(e,exact(p)[0])
            self.assertAlmostEqual(e,p.energy(s))

    def test_invalid_hamiltonian_rejected(self):
        with self.assertRaises(ValueError):Ising([[0,1],[0,0]],[0,0])
        with self.assertRaises(ValueError):Ising([[1,0],[0,0]],[0,0])


class MeasurementTests(unittest.TestCase):
    def test_causal_filter_does_not_read_future(self):
        t=np.arange(1000)*.1;x=np.sin(t)
        a=causal_residual(t,x);x[600:]+=100
        b=causal_residual(t,x)
        np.testing.assert_array_equal(a[:600],b[:600])

    def test_calibration_has_no_test_leakage(self):
        t=np.arange(1000)*.1;x=np.sin(t)
        r=Record('test','test','sha',x,-np.ones(1000),t,{})
        p=prepare_replay(r)
        x[400:]+=50;q=prepare_replay(r)
        self.assertEqual(p['calibration_scale_A'],q['calibration_scale_A'])
        self.assertEqual(p['calibration_center_A'],q['calibration_center_A'])
        self.assertEqual(len(p['test_z']),600)

    def test_reader_uses_voltage_not_sheet_name_and_rejects_time_reset(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'sample.xlsx'
            w=openpyxl.Workbook();s=w.active;s.title='negative-name'
            s.append(['Time','AI','AV'])
            for k in range(4):s.append([k*.1,k*.01,.25])
            w.save(path)
            r=load_records(td)[0]
            np.testing.assert_array_equal(r.voltage,np.full(4,.25))
            s['A4']=0;w.save(path)
            with self.assertRaises(ValueError):load_records(td)


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.p=maxcut(np.array([[0,1,2],[1,0,3],[2,3,0]]))
        self.z=np.random.default_rng(9).normal(size=250)

    def test_modes_report_actual_energy_and_budget(self):
        for mode in MODES:
            s=solve(self.p,self.z,mode,seed=12)
            self.assertAlmostEqual(self.p.energy(s.spins),s.best_energy)
            self.assertGreaterEqual(s.best_energy,exact(self.p)[0]-1e-8)
            self.assertTrue(np.all(np.diff(s.best_history)<=1e-8))
            self.assertEqual(len(s.energy_history),251)
            self.assertEqual(s.samples_consumed,250 if mode in ('replay','shuffled') else 0)

    def test_gain_zero_is_same_greedy_for_all_modes(self):
        reference=solve(self.p,self.z,'greedy',seed=55)
        for mode in MODES:
            s=solve(self.p,self.z,mode,seed=55,gain=0)
            np.testing.assert_array_equal(s.energy_history,reference.energy_history)

    def test_repeatable_and_no_signal_recycling(self):
        a=solve(self.p,self.z,'replay',seed=6);b=solve(self.p,self.z,'replay',seed=6)
        np.testing.assert_array_equal(a.energy_history,b.energy_history)
        self.assertEqual(a.samples_consumed,len(self.z))
        with self.assertRaises(ValueError):solve(self.p,[1,np.nan])
        with self.assertRaises(ValueError):solve(self.p,self.z,gain=-1)

    def test_equal_marginals_in_order_ablation(self):
        a=solve(self.p,self.z,'replay');b=solve(self.p,self.z,'shuffled')
        envelope=np.linspace(1,0,len(self.z))**2
        # Last sample has zero envelope; compare against independent expected permutation.
        expected=np.abs(np.random.default_rng(1000003).permutation(self.z))
        factor=a.thresholds[0]/abs(self.z[0])
        np.testing.assert_allclose(b.thresholds, factor*envelope*expected)


if __name__=='__main__':unittest.main()
