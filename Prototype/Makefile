.PHONY: test resume-last snapshot-session snapshot-auto

test:
	./run_tests.sh

resume-last:
	python session_resume.py resume --last

snapshot-session:
	@test -n "$(TITLE)" || (echo "TITLE is required" && exit 1)
	@test -n "$(SUMMARY)" || (echo "SUMMARY is required" && exit 1)
	@test -n "$(NEXT)" || (echo "NEXT is required" && exit 1)
	python session_resume.py snapshot --title "$(TITLE)" --summary "$(SUMMARY)" --next-step "$(NEXT)" --notes "$(NOTES)"

snapshot-auto:
	python session_resume.py snapshot-auto
