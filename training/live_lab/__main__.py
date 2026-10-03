"""`python -m live_lab`: o mesmo que `live-lab`, sem o .exe que o Smart App Control bloqueia."""

import sys

from live_lab.cli import main

sys.exit(main())
