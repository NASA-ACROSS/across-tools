# Changelog

## [1.3.2](https://github.com/NASA-ACROSS/across-tools/compare/v1.3.1...v1.3.2) (2026-02-27)


### Bug Fixes

* **ci/cd:** make build documentation workflow faster with caching ([#133](https://github.com/NASA-ACROSS/across-tools/issues/133)) ([2233303](https://github.com/NASA-ACROSS/across-tools/commit/2233303f7f56e6e399b0b91d86807e819ddc61dd))
* **visibility:** add `min_vis` support to `compute_joint_visibility` ([#137](https://github.com/NASA-ACROSS/across-tools/issues/137)) ([efd0096](https://github.com/NASA-ACROSS/across-tools/commit/efd0096a36302daf248457925d10067a0ef3c996))
* **visibility:** ensure dates are in whole unites of step_size ([87cd318](https://github.com/NASA-ACROSS/across-tools/commit/87cd3184909e226e6f9a45bd1e604a741ceec66b))
* **visibility:** ensure dates are in whole units of `step_size` ([#135](https://github.com/NASA-ACROSS/across-tools/issues/135)) ([87cd318](https://github.com/NASA-ACROSS/across-tools/commit/87cd3184909e226e6f9a45bd1e604a741ceec66b))

## [1.3.1](https://github.com/NASA-ACROSS/across-tools/compare/v1.3.0...v1.3.1) (2026-02-25)


### Bug Fixes

* **tests:** fix errors in `pytest` caused by caching ([#119](https://github.com/NASA-ACROSS/across-tools/issues/119)) ([e331a3a](https://github.com/NASA-ACROSS/across-tools/commit/e331a3ade56bde7a7bd6fc83ab401a8ba98ac37f))
* **tests:** refactor constraint test directory layout ([#127](https://github.com/NASA-ACROSS/across-tools/issues/127)) ([96e9812](https://github.com/NASA-ACROSS/across-tools/commit/96e981247604178d3c3d136c01c83aadfefaa3ec))
* **visibility:** report correct observatory id in joint visibility ([#130](https://github.com/NASA-ACROSS/across-tools/issues/130)) ([84ebf3f](https://github.com/NASA-ACROSS/across-tools/commit/84ebf3fe8720b9b5454f813c227e23683f0ba626))

## [1.3.0](https://github.com/NASA-ACROSS/across-tools/compare/v1.2.1...v1.3.0) (2026-02-23)


### Features

* **visibility:** add pointing constraint and survey visibility ([#101](https://github.com/NASA-ACROSS/across-tools/issues/101)) ([176f94a](https://github.com/NASA-ACROSS/across-tools/commit/176f94a1f54d2ebe02f73ecad5fb8a97b4ac8d69))


### Bug Fixes

* back compat fix ([ccf3a59](https://github.com/NASA-ACROSS/across-tools/commit/ccf3a5924b53255649618198be9765ee995f433a))
* **constraints:** typing issues with constraints ([#124](https://github.com/NASA-ACROSS/across-tools/issues/124)) ([ccf3a59](https://github.com/NASA-ACROSS/across-tools/commit/ccf3a5924b53255649618198be9765ee995f433a))
* **visibility:** indexing bug with `JointVisibility` for boundary at end of ephemeris ([#125](https://github.com/NASA-ACROSS/across-tools/issues/125)) ([8e271b2](https://github.com/NASA-ACROSS/across-tools/commit/8e271b2d71675f2236c3c8577a8d703a04bc37c3))

## [1.2.1](https://github.com/NASA-ACROSS/across-tools/compare/v1.2.0...v1.2.1) (2026-02-17)


### Bug Fixes

* **visibility:** allow `computed_values` from `Visibility` to be serialized to JSON ([#115](https://github.com/NASA-ACROSS/across-tools/issues/115)) ([9ea2aeb](https://github.com/NASA-ACROSS/across-tools/commit/9ea2aeb43bccc8f71ad3fdea13cbe2d44e91eb5b))
* **visibility:** fix crash in `SolarSystemConstraint` ([#117](https://github.com/NASA-ACROSS/across-tools/issues/117)) ([044a340](https://github.com/NASA-ACROSS/across-tools/commit/044a34068b4073b37ecb264082c0a8c763f745e6))

## [1.2.0](https://github.com/NASA-ACROSS/across-tools/compare/v1.1.0...v1.2.0) (2026-02-17)


### Features

* add xor to constraint operators ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* **footprint:** Add footprint.contains(coordinate) method ([#103](https://github.com/NASA-ACROSS/across-tools/issues/103)) ([054f03d](https://github.com/NASA-ACROSS/across-tools/commit/054f03d7840da4f674a613b11ec26e08f50f87a0))
* **footprint:** adding plotting functionality for footprints ([#92](https://github.com/NASA-ACROSS/across-tools/issues/92)) ([c875b6a](https://github.com/NASA-ACROSS/across-tools/commit/c875b6a3c3892a0656dead5fccacba1dbf68115b))
* **footprint:** query_pixel to use mocpy instead of healpy ([#94](https://github.com/NASA-ACROSS/across-tools/issues/94)) ([dc4ca81](https://github.com/NASA-ACROSS/across-tools/commit/dc4ca81db3da51adad28b9e4e6b70f45c8ac773f))
* **visibility:** Add additional constraints to visibility calculator ([#105](https://github.com/NASA-ACROSS/across-tools/issues/105)) ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* **visibility:** allow visibility constraints to record calculated values in output ([#100](https://github.com/NASA-ACROSS/across-tools/issues/100)) ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))
* **visibility:** combine visibility constraints with logical operators ([#98](https://github.com/NASA-ACROSS/across-tools/issues/98)) ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* **visibility:** record computed visibility values in result ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))


### Bug Fixes

* actually commit bandpass update ([d2a5466](https://github.com/NASA-ACROSS/across-tools/commit/d2a546679433db295bba650f249a265dbca3603b))
* add pass through of computed values in joint and composite constraints ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* add pass through of computed values in joint and composite constraints ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))
* add test coverage to constructor ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* add xor to constraint abc docstring ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* allow Constraint serializer to take single constraint ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* bad import in test ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* **bandpass:** add validation on bandwidth and central_wavelength ([#110](https://github.com/NASA-ACROSS/across-tools/issues/110)) ([d2da967](https://github.com/NASA-ACROSS/across-tools/commit/d2da96776b21240ecd06c2af78aca1cc9501279c))
* clean up imports in unit test ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* clean up imports in unit test ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))
* docstring and method update ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* docstring and method update ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))
* **docs:** update documentation missing from recent PRs and fix docstrings ([#109](https://github.com/NASA-ACROSS/across-tools/issues/109)) ([d2a5466](https://github.com/NASA-ACROSS/across-tools/commit/d2a546679433db295bba650f249a265dbca3603b))
* **ephemeris:** optimize TLE ephemeris generation using `rust_ephem` ([#96](https://github.com/NASA-ACROSS/across-tools/issues/96)) ([9b5831a](https://github.com/NASA-ACROSS/across-tools/commit/9b5831a6138bbfdc75c6f23fd773a22878cca201))
* hacky fix for serialization of combined constraints ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* issue where constraint type of a combined constraint can return unknown ([#114](https://github.com/NASA-ACROSS/across-tools/issues/114)) ([63bd49e](https://github.com/NASA-ACROSS/across-tools/commit/63bd49e48ba86fc4983f54854699ea7ee1745325))
* linting and mypy fixes ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* linting and mypy fixes ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))
* mypy errors in unit tests ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* mypy issues with unit tests ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* reduce boilerplate with mixin ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* reinstate missing function ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* reinstate missing function ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))
* reinstate missing tests ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* reinstate missing tests ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))
* remove tech spec document ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* remove tech spec document ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))
* show correct violation in constraint ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* simplify _merge_computed_values in ephemeris_visibility ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* simplify _merge_computed_values in ephemeris_visibility ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))
* slightly less hacky pydantic ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* slightly less hacky pydantic ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))
* slightly less hacky, but still a bit hacky ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* support passing single constraint to EphemerisVisibility. Fix constraint reporting. ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* unit tests ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* up test coverage ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* update API docs ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* update doc example ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* update docstrings ([f260446](https://github.com/NASA-ACROSS/across-tools/commit/f260446ef29d6fb717ebd3ec92739d7329bb2286))
* update google style docstrings to numpy ([d2a5466](https://github.com/NASA-ACROSS/across-tools/commit/d2a546679433db295bba650f249a265dbca3603b))
* update some out of date docstrings ([d2a5466](https://github.com/NASA-ACROSS/across-tools/commit/d2a546679433db295bba650f249a265dbca3603b))
* **visibility:** correct computed_values ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* **visibility:** correct computed_values ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))


### Documentation

* add draft tech spec ([ce180f8](https://github.com/NASA-ACROSS/across-tools/commit/ce180f8932da212302ef4e55be31d1e8e0a5bd9f))
* add draft tech spec ([c12d74c](https://github.com/NASA-ACROSS/across-tools/commit/c12d74cf0cc6c3c8019fc1d75191efcbec5e1889))

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
