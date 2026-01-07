# Changelog

## [1.1.0](https://github.com/NASA-ACROSS/across-tools/compare/v1.0.0...v1.1.0) (2026-01-07)


### Features

* **actions:** add support for release-please ([#87](https://github.com/NASA-ACROSS/across-tools/issues/87)) ([48faa23](https://github.com/NASA-ACROSS/across-tools/commit/48faa23d5ef31ad8d020e26197ce2c740194027b))
* **actions:** test on python 3.13 and 3.14 ([#77](https://github.com/NASA-ACROSS/across-tools/issues/77)) ([6fdd3a5](https://github.com/NASA-ACROSS/across-tools/commit/6fdd3a51b49b89be7ec7c909a368b0fdcded2cfe))
* add functionality to calculate joint visibility windows ([#55](https://github.com/NASA-ACROSS/across-tools/issues/55)) ([ab97937](https://github.com/NASA-ACROSS/across-tools/commit/ab97937e4988732c8d3c1b7733c2e2deb364de61))
* Add saa unit tests ([45eb6ec](https://github.com/NASA-ACROSS/across-tools/commit/45eb6ecee7832dc9706281f87330feee2bb5be54))
* **deps:** pin lower bounds on dependencies to make sure module works well ([#83](https://github.com/NASA-ACROSS/across-tools/issues/83)) ([2124465](https://github.com/NASA-ACROSS/across-tools/commit/21244651a5377b770b07db200a26b11b6f85b703))
* **docs:** add API documentation ([#66](https://github.com/NASA-ACROSS/across-tools/issues/66)) ([1cf246a](https://github.com/NASA-ACROSS/across-tools/commit/1cf246ab747688a7c5f1ffe6b82b26dbd2eb8d69))
* **ephemeris:** Create Ephemeris Calculation Tool ([#16](https://github.com/NASA-ACROSS/across-tools/issues/16)) ([603ad61](https://github.com/NASA-ACROSS/across-tools/commit/603ad61cda8eb810b1fecc2d21fef190a8f3f8ed))
* **footprint:** Adding footprint analysis tools ([#12](https://github.com/NASA-ACROSS/across-tools/issues/12)) ([65dafea](https://github.com/NASA-ACROSS/across-tools/commit/65dafea8990e4cac274b156ad32110814e315927))
* **legal:** add NASA copyright notices ([#70](https://github.com/NASA-ACROSS/across-tools/issues/70)) ([21883ef](https://github.com/NASA-ACROSS/across-tools/commit/21883efdbd3c271c1e280f5e68dfcf4b49fd2a9a))
* **README:** installation instructions with pypi ([#89](https://github.com/NASA-ACROSS/across-tools/issues/89)) ([d45354b](https://github.com/NASA-ACROSS/across-tools/commit/d45354b22c40942a82899082dfb85932f020216e))
* **TLE:** Add code to fetch TLE from space-track.org ([#23](https://github.com/NASA-ACROSS/across-tools/issues/23)) ([2c72f1f](https://github.com/NASA-ACROSS/across-tools/commit/2c72f1f7d385b17170fb0960f05ba5e50f874005))
* **tools:** Changed namespace: src to across ([#7](https://github.com/NASA-ACROSS/across-tools/issues/7)) ([a01cb6a](https://github.com/NASA-ACROSS/across-tools/commit/a01cb6a865ab99b09bbbf0126d105fdb8429df55))
* Update dependencies for pypi uploads ([#79](https://github.com/NASA-ACROSS/across-tools/issues/79)) ([83a84a2](https://github.com/NASA-ACROSS/across-tools/commit/83a84a2b804d153b6156112db6ced72066bed1f7))
* Update Readme ([#64](https://github.com/NASA-ACROSS/across-tools/issues/64)) ([c6ccf1b](https://github.com/NASA-ACROSS/across-tools/commit/c6ccf1bc9d80b17d47107c84b9fd5f28ab44e5f2))
* **visibility:** Add Ephemeris based visibility calculator ([#33](https://github.com/NASA-ACROSS/across-tools/issues/33)) ([9baac59](https://github.com/NASA-ACROSS/across-tools/commit/9baac59e6b6138b3a9d3ed1600219dcf1ad6c99f))
* **visibility:** Visibility calculator for polygon constraints ([#40](https://github.com/NASA-ACROSS/across-tools/issues/40)) ([45eb6ec](https://github.com/NASA-ACROSS/across-tools/commit/45eb6ecee7832dc9706281f87330feee2bb5be54))
* **visibiliy:** Add polygon based constraints ([45eb6ec](https://github.com/NASA-ACROSS/across-tools/commit/45eb6ecee7832dc9706281f87330feee2bb5be54))


### Bug Fixes

* alt-az definition and tests ([45eb6ec](https://github.com/NASA-ACROSS/across-tools/commit/45eb6ecee7832dc9706281f87330feee2bb5be54))
* **build:** Define `mypy` strict mode globally ([#18](https://github.com/NASA-ACROSS/across-tools/issues/18)) ([eebab2a](https://github.com/NASA-ACROSS/across-tools/commit/eebab2a03ab520ac59ad39a538dfbd55d9f0c647))
* **build:** Define mypy strict mode globally ([eebab2a](https://github.com/NASA-ACROSS/across-tools/commit/eebab2a03ab520ac59ad39a538dfbd55d9f0c647))
* **constraints:** making constraint polygon nullable ([#57](https://github.com/NASA-ACROSS/across-tools/issues/57)) ([8358250](https://github.com/NASA-ACROSS/across-tools/commit/835825092e8fd08d1952d6205d37d6c07cefa46a))
* **ephemeris:** Add missing `latitude`/`longitude`/`height` attributes to `TLEEphemeris` ([#37](https://github.com/NASA-ACROSS/across-tools/issues/37)) ([34e5573](https://github.com/NASA-ACROSS/across-tools/commit/34e5573cae75394ce01ba187693b79f1db5b0985))
* **ephemeris:** Round begin/end to step_size for consistency ([#48](https://github.com/NASA-ACROSS/across-tools/issues/48)) ([cd87d0a](https://github.com/NASA-ACROSS/across-tools/commit/cd87d0aadc7ec37bee5894efcd80ce896f08a5e8))
* **license:** change ACROSS Team to NASA ACROSS ([#75](https://github.com/NASA-ACROSS/across-tools/issues/75)) ([486dba3](https://github.com/NASA-ACROSS/across-tools/commit/486dba369643321c54826fed36418c02c177b686))
* making constraint polygon nullable ([8358250](https://github.com/NASA-ACROSS/across-tools/commit/835825092e8fd08d1952d6205d37d6c07cefa46a))
* missing arguments in test ([45eb6ec](https://github.com/NASA-ACROSS/across-tools/commit/45eb6ecee7832dc9706281f87330feee2bb5be54))
* **namespace:** add code to ensure namespace works ([#81](https://github.com/NASA-ACROSS/across-tools/issues/81)) ([ec95496](https://github.com/NASA-ACROSS/across-tools/commit/ec9549663c6027d82ab09e094d910da4dae311cb))
* **org-name:** update organization name ([#72](https://github.com/NASA-ACROSS/across-tools/issues/72)) ([434a5a3](https://github.com/NASA-ACROSS/across-tools/commit/434a5a3afdb95d20ef81a1b0616ad0936562d52b))
* **python:** Update code for minimum Python version of 3.10 ([#28](https://github.com/NASA-ACROSS/across-tools/issues/28)) ([e9ba208](https://github.com/NASA-ACROSS/across-tools/commit/e9ba208c57eab80df9d74d90b44c76d641708162))
* **python:** Update code for Python 3.10 minimum ([e9ba208](https://github.com/NASA-ACROSS/across-tools/commit/e9ba208c57eab80df9d74d90b44c76d641708162))
* Remove async methods for now ([2c72f1f](https://github.com/NASA-ACROSS/across-tools/commit/2c72f1f7d385b17170fb0960f05ba5e50f874005))
* **test:** Remove line of code that can never be covered by test ([603ad61](https://github.com/NASA-ACROSS/across-tools/commit/603ad61cda8eb810b1fecc2d21fef190a8f3f8ed))
* **tests:** Add tests to up coverage to 100% ([#20](https://github.com/NASA-ACROSS/across-tools/issues/20)) ([9f80952](https://github.com/NASA-ACROSS/across-tools/commit/9f809529d5324e42799e14b4be4cb1299924bb64))
* **tests:** Add unit tests for TLE fetch ([2c72f1f](https://github.com/NASA-ACROSS/across-tools/commit/2c72f1f7d385b17170fb0960f05ba5e50f874005))
* **tests:** Refactor for only one assert per test ([2c72f1f](https://github.com/NASA-ACROSS/across-tools/commit/2c72f1f7d385b17170fb0960f05ba5e50f874005))
* **tests:** Split up a test ([2c72f1f](https://github.com/NASA-ACROSS/across-tools/commit/2c72f1f7d385b17170fb0960f05ba5e50f874005))
* **tests:** Tests at 100% coverage ([2c72f1f](https://github.com/NASA-ACROSS/across-tools/commit/2c72f1f7d385b17170fb0960f05ba5e50f874005))
* **tle:** Remove test that isn't reachable ([2c72f1f](https://github.com/NASA-ACROSS/across-tools/commit/2c72f1f7d385b17170fb0960f05ba5e50f874005))
* **tle:** replace soon deprecated space-track endpoint `/tle` with `/gp_history` ([#85](https://github.com/NASA-ACROSS/across-tools/issues/85)) ([b1ee5fb](https://github.com/NASA-ACROSS/across-tools/commit/b1ee5fb791346d0d8c57b6b715f7dbdf2651f08d))
* **typing:** Removed type:ignore that was causing mypy to thrown error in CI ([#35](https://github.com/NASA-ACROSS/across-tools/issues/35)) ([46eae53](https://github.com/NASA-ACROSS/across-tools/commit/46eae53958e3d7a57537fd4e220c311dddf072c1))
* Update docstrings ([2c72f1f](https://github.com/NASA-ACROSS/across-tools/commit/2c72f1f7d385b17170fb0960f05ba5e50f874005))
* Update method to match draft ticket ([2c72f1f](https://github.com/NASA-ACROSS/across-tools/commit/2c72f1f7d385b17170fb0960f05ba5e50f874005))
* Update TLE code to target Python 3.9 ([2c72f1f](https://github.com/NASA-ACROSS/across-tools/commit/2c72f1f7d385b17170fb0960f05ba5e50f874005))
* **visibility:** add `min_vis` argument to `compute_ephemeris_visibility` ([#42](https://github.com/NASA-ACROSS/across-tools/issues/42)) ([80dc179](https://github.com/NASA-ACROSS/across-tools/commit/80dc1791c2cbc230f0b74622cda4379c07b111b8))
* **visibility:** add min_vis argument to compute_ephemeris_visibility ([80dc179](https://github.com/NASA-ACROSS/across-tools/commit/80dc1791c2cbc230f0b74622cda4379c07b111b8))
* **visibility:** Fix enum representations appearing in `constraint_reason` ([#44](https://github.com/NASA-ACROSS/across-tools/issues/44)) ([b75cb3e](https://github.com/NASA-ACROSS/across-tools/commit/b75cb3e080d2f2d49989d633bb2801a261124e4a))
* **visibility:** Fix enum representations appearing in constraint_reason ([b75cb3e](https://github.com/NASA-ACROSS/across-tools/commit/b75cb3e080d2f2d49989d633bb2801a261124e4a))
* **visibility:** fix issue instantiating `SAAPolygonConstraint` from JSON ([#46](https://github.com/NASA-ACROSS/across-tools/issues/46)) ([1bfa5cb](https://github.com/NASA-ACROSS/across-tools/commit/1bfa5cb0bdee2afd8a040f5f3defde0b7350471e))


### Documentation

* remove .github readme ([#13](https://github.com/NASA-ACROSS/across-tools/issues/13)) ([b80dec7](https://github.com/NASA-ACROSS/across-tools/commit/b80dec79d980c6f175abe87ff9bc83fa5f7ad24e))
* rename bug.md to bug.yaml ([b211af6](https://github.com/NASA-ACROSS/across-tools/commit/b211af67d0f9fa18d1a6dbefd1ff3fde80ae6ef3))
* rename spike.md to spike.yaml ([c457544](https://github.com/NASA-ACROSS/across-tools/commit/c457544aff589b6a95f66fed7f6ef769eecd7dcb))
* rename ticket.md to ticket.yaml ([3e2b109](https://github.com/NASA-ACROSS/across-tools/commit/3e2b109da258f42f4ad54b3b0d411b17462eb76e))
* update bug.md to follow yaml syntax ([c5107a8](https://github.com/NASA-ACROSS/across-tools/commit/c5107a8f8367f344cf95c387472252ec12d186e2))
* update pull_request_template.md and ISSUE_TEMPLATES ([#15](https://github.com/NASA-ACROSS/across-tools/issues/15)) ([371236d](https://github.com/NASA-ACROSS/across-tools/commit/371236db1c844380c704fd9ca73d58890c043db6))
* update spike.md template to follow yaml syntax ([d24565f](https://github.com/NASA-ACROSS/across-tools/commit/d24565f16cae9a61c880ccaa1d5f3cbd8dea370c))
* update ticket.md template to follow yaml syntax ([3326ce1](https://github.com/NASA-ACROSS/across-tools/commit/3326ce120ebbee6c7f7a88adc6ef196e26e54362))
