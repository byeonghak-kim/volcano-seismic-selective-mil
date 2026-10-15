# Data interface

`events_manifest.csv` contains one row per catalogue event. `event_time` is the
first-record time used for partitioning. Both record columns contain stable keys
for physical station-minute records; repeated keys are intentional and create
overlap groups. Labels must be `A`, `B`, or `T`.

`records_manifest.csv` maps each record key to one relative `.npy` path and its
original sampling rate. Every array has shape `(3, samples)` with component order
E, N, U/Z and contains finite SAC samples in stored/native units. Paths are
resolved beneath `--records-root`; absolute paths and parent traversal are
rejected. The public package does not require or encode institutional directory
names, station aliases, catalogue identifiers, or absolute paths.

Before manifest construction, collapse upstream catalogue subcodes as used in
the study: BH/BL/BP/BS/BT to `B`, and T(c)/T(e)/T(s)/TK/TP to `T`.
