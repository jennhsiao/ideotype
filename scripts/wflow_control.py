"""Generate init text files for CONTROL maizsim simulations."""

from ideotype.log import log_fetchinfo
from ideotype.wflow_setup import (make_dircts, make_inits, make_cultivars,
                                  make_runs, make_jobs, make_subjobs)

log_fetchinfo('control')
make_dircts('control', cont_cvars=False)
make_inits('control')
make_cultivars('control', cont_cvars=False)
make_runs('control', cont_cvars=False)
make_jobs('control', cont_cvars=False)
make_subjobs('control')
