from pathlib import Path
import csv
import json
import unittest
import numpy as np
from measurement_analysis.io import load_records, sha256
from measurement_analysis.metrics import causal_residual, summarize, iv_branches, windows

ROOT = Path(__file__).resolve().parents[1]


class Measurements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records(ROOT)

    def test_original_files_match_manifest(self):
        manifest = json.loads((ROOT/'analysis/file_manifest.json').read_text())
        self.assertEqual(len(manifest), 21)
        for item in manifest:
            self.assertEqual(sha256(ROOT/item['path']), item['sha256'], item['path'])

    def test_complete_record_inventory(self):
        for dataset, count, it_count in [('original',3,3),('r4c5',49,25),('r4c4',21,21)]:
            subset = [r for r in self.records if r.dataset == dataset]
            self.assertEqual(len(subset), count)
            self.assertEqual(sum(r.time is not None for r in subset), it_count)
        self.assertEqual(sum(len(r.current) for r in self.records), 202432)

    def test_missing_voltage_is_not_imputed(self):
        record = next(r for r in self.records if r.source.endswith('/random number.xlsx'))
        self.assertIsNone(record.voltage)
        self.assertIsNone(summarize(record)['voltage_median_V'])
        self.assertEqual(len(record.current),4096)

    def test_voltage_comes_from_column_not_sheet_name(self):
        r = next(r for r in self.records if r.sheet=='160R4C5-0.25VA1V')
        self.assertAlmostEqual(summarize(r)['voltage_median_V'], .25)

    def test_light_device_is_separate(self):
        new = [r for r in self.records if r.dataset=='r4c4']
        self.assertEqual(sum(r.device=='R4C4' for r in new),17)
        self.assertEqual({r.sheet for r in new if r.device=='Light_other_device'},
                         {'Run3179','Run3180','Run3181','Run3182'})

    def test_causal_processing_does_not_read_future_samples(self):
        t = np.arange(100)*.125
        first = np.sin(t)
        changed = first.copy(); changed[60:] += 100
        np.testing.assert_array_equal(causal_residual(t,first)[:60], causal_residual(t,changed)[:60])

    def test_previously_computed_reference_values(self):
        reference = {'Run3170':83.64252139778074,'Run3172':1001.923204036137,
                     'Run3184':837.2345130020915,'Run3185':31.911760955519327}
        for sheet,value in reference.items():
            r = next(r for r in self.records if r.sheet==sheet)
            self.assertAlmostEqual(summarize(r)['residual_sigma_uA'],value,places=6)
        r = next(r for r in self.records if r.sheet=='Run3185')
        late = next(w for w in windows(r) if w['window']=='late')
        self.assertAlmostEqual(late['residual_sigma_uA'],5.4210,places=3)

    def test_iv_branch_rows_and_statistics(self):
        records = [r for r in self.records if r.dataset=='r4c5' and r.time is None]
        branches = [b for r in records for b in iv_branches(r)]
        self.assertEqual(len(branches),48)
        for voltage, reference in [(.5,.27077348743538066),(-.5,.24852533463910742)]:
            actual = np.median([b['relative_difference'] for b in branches if b['voltage_V']==voltage])
            self.assertAlmostEqual(actual,reference,places=10)
        for r in records:
            for b in iv_branches(r):
                self.assertAlmostEqual(r.current[b['first_row']-2]*1e3,b['first_mA'])

    def test_archived_summary_matches_raw_records(self):
        with (ROOT/'analysis/all_measurements.csv').open() as handle:
            archived = {(r['source'],r['sheet']):r for r in csv.DictReader(handle)}
        self.assertEqual(len(archived),len(self.records))
        for r in self.records:
            a = summarize(r); saved = archived[(r.source,r.sheet)]
            self.assertAlmostEqual(float(saved['current_mean_mA']),a['current_mean_mA'],places=10)
            if r.time is not None:
                self.assertAlmostEqual(float(saved['residual_sigma_uA']),a['residual_sigma_uA'],places=8)


if __name__ == '__main__':
    unittest.main()
