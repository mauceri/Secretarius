#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from secretarius.email_adapter import EmailPollingAdapter


def main() -> int:
    adapter = EmailPollingAdapter()
    adapter.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

