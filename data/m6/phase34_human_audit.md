# Phase 34 人工审核表：workspace 词 vs 文档关联度

判断标准：每个 workspace 词是否与当前文档内容相关？

- R = 相关（文档讨论的概念，包括总结性但正确的词如 oncology）
- G = 泛泛（领域联想但文档没具体讨论，如看到 cancer 就想到 tumor）
- N = 噪声（artifact/无关词）


## cancer + chemotherapy
生成: "that chemotherapy is a common treatment used to kill cancer cells and shrink tumors, often prescribe"

| step | gen_token | workspace top-5 (prob) | 判断 |
|------|-----------|------------------------|------|
| 0 | that | therapeutic(0.71) causal(0.02) apeut(0.01) medicinal(0.00) itational(0.00) | |
| 1 | chemotherapy | chemotherapy(0.98) chem(0.00) cancer(0.00) chemistry(0.00) doctors(0.00) | |
| 2 | is | drugs(0.02) refers(0.01) elmet(0.01) ulings(0.01) agini(0.01) | |
| 3 | a | used(0.23) prescribed(0.03) commonly(0.01) ulings(0.01) employed(0.01) | |
| 4 | common | common(0.14) primary(0.03) type(0.02) cancer(0.01) therapeutic(0.01) | |
| 5 | treatment | cancer(0.01) therapeutic(0.00) medication(0.00) type(0.00) therapy(0.00) | |
| 6 | used | used(0.23) approach(0.09) prescribed(0.05) modal(0.02) regimen(0.02) | |
| 7 | to | asurer(0.03) циально(0.03) ntag(0.02) medically(0.01) amac(0.01) | |
| 8 | kill | kill(0.67) treat(0.03) eliminate(0.02) shrink(0.01) combat(0.01) | |
| 9 | cancer | cancer(0.75) rapidly(0.00) tumor(0.00) rapid(0.00) malignant(0.00) | |
| 10 | cells | cells(0.79) cell(0.00) ocytes(0.00) uous(0.00) ious(0.00) | |
| 11 | and | aniu(0.06) acje(0.06) entai(0.02) urnished(0.02) otre(0.02) | |
| 12 | shrink | shrink(0.84) shrinking(0.01) manage(0.01) reduce(0.01) prevent(0.00) | |
| 13 | tumors | tumors(0.98) tumor(0.01) cancer(0.00) malignant(0.00) cancers(0.00) | |
| 14 | , | anyl(0.02) inati(0.02) ignet(0.01) elige(0.01) aniu(0.01) | |
| 15 | often | urnished(0.09) often(0.09) ungeons(0.03) administered(0.03) prescribed(0.02) | |
| 16 | prescribed | prescribed(0.97) prescribe(0.00) prescription(0.00) pres(0.00) prescribing(0.00) | |
| 17 | by | licted(0.00) agini(0.00) ylko(0.00) odyn(0.00) циально(0.00) | |
| 18 | medical | medically(0.00) hemat(0.00) cancer(0.00) medial(0.00) specialists(0.00) | |
| 19 | onc | specialists(0.00) doctors(0.00) professionals(0.00) tumor(0.00) cancer(0.00) | |
| 20 | ologists | ologists(1.00) ologist(0.00) ology(0.00) ists(0.00) ographers(0.00) | |
| 21 | . | errated(0.01) specializing(0.01) rray(0.01) ierz(0.01) aniem(0.00) | |
| 22 | <|im_end|> | aniu(0.00) kich(0.00) zych(0.00) acje(0.00) owane(0.00) | |

**所有领域词**: acje, actively, administered, afür, against, agini, aiming, aised, akit, alleviate, amac, ancer, ancers, aniem, aniu, antic, antically, anyl, apeut, apeutic
**关系词**: caus, causal, critical, crucial, guided, kills, prevent, reatment, therapeutic, therapeutic


## cancer + surgery
生成: "that surgery is a common treatment method used to remove cancerous tumors, including those of basal "

| step | gen_token | workspace top-5 (prob) | 判断 |
|------|-----------|------------------------|------|
| 0 | that | often(0.26) evident(0.13) apeut(0.11) therapeutic(0.07) illustrated(0.01) | |
| 1 | surgery | surgery(0.98) surgical(0.01) cancer(0.00) surgeries(0.00) surgeons(0.00) | |
| 2 | is | often(0.03) serves(0.00) commonly(0.00) ivor(0.00) ustum(0.00) | |
| 3 | a | commonly(0.46) often(0.32) frequently(0.01) common(0.00) часто(0.00) | |
| 4 | common | common(0.81) primary(0.00) commonly(0.00) prevalent(0.00) typical(0.00) | |
| 5 | treatment | therapeutic(0.01) surgical(0.00) reatment(0.00) treat(0.00) approach(0.00) | |
| 6 | method | approach(0.24) modal(0.03) option(0.02) methodology(0.00) procedure(0.00) | |
| 7 | used | used(0.31) aimed(0.01) utilized(0.01) ustum(0.00) utilisé(0.00) | |
| 8 | to | eldom(0.00) esub(0.00) agnost(0.00) okit(0.00) циально(0.00) | |
| 9 | remove | remove(0.94) surg(0.00) removes(0.00) removal(0.00) removed(0.00) | |
| 10 | cancer | cancer(0.89) tumors(0.07) cancers(0.01) tumor(0.01) malignant(0.00) | |
| 11 | ous | cells(0.00) tumors(0.00) tissues(0.00) tissue(0.00) orous(0.00) | |
| 12 | tumors | tumors(0.70) tissues(0.19) tissue(0.06) cells(0.02) lesions(0.01) | |
| 13 | , | inic(0.00) ksam(0.00) peare(0.00) prus(0.00) ardu(0.00) | |
| 14 | including | ylko(0.04) okit(0.00) aiming(0.00) particularly(0.00) ierz(0.00) | |
| 15 | those | basal(0.96) those(0.03) cases(0.00) cancers(0.00) baseline(0.00) | |
| 16 | of | seen(0.05) elmet(0.03) cases(0.02) caused(0.02) types(0.02) | |
| 17 | basal | basal(1.00) baseline(0.00) basil(0.00) skin(0.00) Basel(0.00) | |
| 18 | cell | cell(0.96) actable(0.00) elmet(0.00) pressão(0.00) melan(0.00) | |
| 19 | skin | skin(0.56) carcinoma(0.43) carcin(0.00) cancer(0.00) melan(0.00) | |
| 20 | cancer | cancer(0.98) carcinoma(0.01) cancers(0.00) carcin(0.00) tumors(0.00) | |
| 21 | . | agini(0.00) ivor(0.00) umbn(0.00) astle(0.00) aniu(0.00) | |
| 22 | <|im_end|> | aniu(0.00) kich(0.00) acje(0.00) nika(0.00) zych(0.00) | |

**所有领域词**: abnormal, abnormalities, acje, actable, affecting, afür, agini, agnost, aimed, aiming, aims, akit, alink, amac, ancell, ancellor, ancer, ancers, aniem, aniu
**关系词**: caused, growth, reatment, removal, removal, therapeutic, therapeutic, therapeutic, therapy, therapy


## smoking + cancer
生成: "that smoking is a significant risk factor for various types of cancer, including those that originat"

| step | gen_token | workspace top-5 (prob) | 判断 |
|------|-----------|------------------------|------|
| 0 | that | causal(0.02) significant(0.02) caus(0.02) osit(0.02) alink(0.01) | |
| 1 | smoking | smoking(0.84) cigarette(0.02) tobacco(0.01) habitual(0.00) prolonged(0.00) | |
| 2 | is | significantly(0.06) tobacco(0.04) increases(0.03) cigarettes(0.03) greatly(0.02) | |
| 3 | a | caus(0.31) linked(0.03) strongly(0.03) ensely(0.01) trak(0.01) | |
| 4 | significant | significant(0.26) leading(0.15) major(0.07) caus(0.05) known(0.02) | |
| 5 | risk | risk(0.42) caus(0.07) causal(0.03) carcin(0.02) contributing(0.02) | |
| 6 | factor | factor(0.95) factors(0.00) фактор(0.00) multiplier(0.00) contributor(0.00) | |
| 7 | for | leading(0.08) contributing(0.06) ribly(0.02) causing(0.01) caus(0.01) | |
| 8 | various | developing(0.84) several(0.01) numerous(0.01) develop(0.01) causing(0.00) | |
| 9 | types | types(0.71) cancers(0.23) forms(0.00) cancer(0.00) kinds(0.00) | |
| 10 | of | cancers(0.05) cancer(0.02) ections(0.00) ardu(0.00) ocache(0.00) | |
| 11 | cancer | cancer(0.96) cancers(0.04) malign(0.00) carcin(0.00) tumors(0.00) | |
| 12 | , | ocache(0.00) ignet(0.00) imary(0.00) emean(0.00) oldem(0.00) | |
| 13 | including | particularly(0.02) especially(0.00) ardu(0.00) notably(0.00) includ(0.00) | |
| 14 | those | cancers(0.96) lung(0.01) cancer(0.01) carcin(0.00) those(0.00) | |
| 15 | that | affecting(0.70) originating(0.11) arising(0.01) occurring(0.00) stemming(0.00) | |
| 16 | originate | originate(0.68) affect(0.10) arise(0.09) occur(0.02) develop(0.02) | |
| 17 | in | within(0.00) epith(0.00) emean(0.00) ections(0.00) presso(0.00) | |
| 18 | the | organs(0.61) tissues(0.25) epith(0.05) lung(0.02) areas(0.01) | |
| 19 | lungs | lungs(0.61) respiratory(0.29) lung(0.08) epith(0.01) tissues(0.00) | |
| 20 | , | abyrin(0.00) rray(0.00) agini(0.00) elmet(0.00) contri(0.00) | |
| 21 | throat | throat(0.31) mouth(0.12) oral(0.04) respiratory(0.04) ulings(0.03) | |
| 22 | , | roat(0.00) owied(0.00) esub(0.00) ogui(0.00) oenix(0.00) | |
| 23 | mouth | mouth(0.91) bladder(0.04) lips(0.01) oral(0.01) mouths(0.00) | |
| 24 | , | oenix(0.00) esub(0.00) ogui(0.00) asurer(0.00) elmet(0.00) | |
| 25 | and | bladder(0.80) kidneys(0.11) pancre(0.03) liver(0.01) kidney(0.00) | |
| 26 | other | bladder(0.22) kidneys(0.07) errated(0.06) others(0.02) lungs(0.02) | |
| 27 | organs | organs(0.56) tissues(0.34) parts(0.02) areas(0.02) respiratory(0.00) | |
| 28 | , | owied(0.02) ivors(0.01) exposed(0.01) rray(0.01) contri(0.01) | |
| 29 | as | astle(0.04) utow(0.03) aniu(0.01) potentially(0.01) primarily(0.01) | |

**所有领域词**: abyrin, achinery, actor, affect, affecting, affects, agini, agnar, ajor, alars, alink, amac, ancer, ancers, aniu, anky, ardu, areas, arise, arising
**关系词**: affects, affects, caus, caus, caus, caus, caus, causal, causal, causal


## tumor + growth
生成: "that a tumor is an abnormal growth of cells, which can occur in the adrenal glands and other parts o"

| step | gen_token | workspace top-5 (prob) | 判断 |
|------|-----------|------------------------|------|
| 0 | that | evident(0.06) utow(0.01) aternity(0.01) osit(0.01) umbnail(0.01) | |
| 1 | a | tumors(0.56) tumor(0.44) cancer(0.00) cancers(0.00) malignant(0.00) | |
| 2 | tumor | tumor(1.00) tumors(0.00) growth(0.00) umor(0.00) cancer(0.00) | |
| 3 | is | represents(0.19) growth(0.16) refers(0.03) grows(0.03) forms(0.01) | |
| 4 | an | abnormal(0.50) growth(0.01) aberr(0.01) often(0.01) abox(0.00) | |
| 5 | abnormal | abnormal(0.96) growth(0.00) aberr(0.00) abnormalities(0.00) anom(0.00) | |
| 6 | growth | growth(0.92) mass(0.00) tissue(0.00) lump(0.00) formation(0.00) | |
| 7 | of | caused(0.04) indered(0.03) gnore(0.01) ustum(0.01) elige(0.01) | |
| 8 | cells | cells(0.54) tissues(0.22) tissue(0.22) cellular(0.00) abnormal(0.00) | |
| 9 | , | incy(0.03) ignet(0.03) rray(0.02) ynos(0.01) imary(0.01) | |
| 10 | which | which(0.38) often(0.04) olicited(0.01) benign(0.00) zych(0.00) | |
| 11 | can | esub(0.03) occurs(0.01) arises(0.00) falls(0.00) occur(0.00) | |
| 12 | occur | occur(0.93) arise(0.01) affect(0.00) develop(0.00) potentially(0.00) | |
| 13 | in | anywhere(0.68) benign(0.02) naturally(0.01) inati(0.01) spontaneously(0.01) | |
| 14 | the | tissues(0.59) glands(0.22) organs(0.09) tissue(0.02) gland(0.01) | |
| 15 | adrenal | adrenal(0.84) glands(0.11) tissues(0.03) tissue(0.01) gland(0.00) | |
| 16 | glands | glands(1.00) gland(0.00) tissues(0.00) tissue(0.00) hormones(0.00) | |
| 17 | and | affecting(0.56) influencing(0.03) impacting(0.01) zyst(0.01) iect(0.01) | |
| 18 | other | affect(0.80) potentially(0.05) lead(0.01) affects(0.01) elsewhere(0.01) | |
| 19 | parts | tissues(0.70) parts(0.09) organs(0.07) areas(0.01) body(0.00) | |
| 20 | of | emean(0.00) escort(0.00) ardu(0.00) intégr(0.00) agini(0.00) | |
| 21 | the | bodily(0.00) reten(0.00) apesh(0.00) body(0.00) ioxide(0.00) | |
| 22 | body | body(1.00) bodies(0.00) bodily(0.00) Körper(0.00) corpo(0.00) | |
| 23 | . | affecting(0.02) alink(0.00) amac(0.00) impacting(0.00) influencing(0.00) | |
| 24 | <|im_end|> | aniu(0.00) kich(0.00) acje(0.00) amac(0.00) ogui(0.00) | |

**所有领域词**: abant, aberr, abnormal, abnormalities, abox, abwe, acje, aden, adip, adrenal, affect, affecting, affects, agini, alink, amac, anatomy, ancers, aniu, anky
**关系词**: affects, cause, caused, causing, growth, growth, growth, growth, growth, growth


## insulin + diabetes
生成: "that diabetes affects the body's ability to produce or respond to insulin, leading to high blood sug"

| step | gen_token | workspace top-5 (prob) | 判断 |
|------|-----------|------------------------|------|
| 0 | that | fundamental(0.04) crucial(0.03) central(0.01) profound(0.01) critical(0.01) | |
| 1 | diabetes | diabetes(0.62) insulin(0.37) diabetic(0.00) insufficient(0.00) individuals(0.00) | |
| 2 | affects | affects(0.93) impacts(0.01) disrupt(0.00) occurs(0.00) interfer(0.00) | |
| 3 | the | insulin(0.29) pancreatic(0.01) pancre(0.00) cells(0.00) glucose(0.00) | |
| 4 | body | body(0.97) pancre(0.02) insulin(0.00) bodies(0.00) regulation(0.00) | |
| 5 | 's | riger(0.01) emean(0.01) icontrol(0.01) emade(0.01) okit(0.01) | |
| 6 | ability | ability(0.95) insulin(0.05) utilization(0.00) regulation(0.00) production(0.00) | |
| 7 | to | regulate(0.00) reon(0.00) rray(0.00) orsche(0.00) ignet(0.00) | |
| 8 | produce | produce(0.72) regulate(0.11) properly(0.10) utilize(0.01) insulin(0.01) | |
| 9 | or | insulin(0.00) adequate(0.00) sufficient(0.00) insufficient(0.00) enough(0.00) | |
| 10 | respond | respond(0.64) properly(0.29) effectively(0.02) correctly(0.01) utilize(0.00) | |
| 11 | to | properly(0.32) appropriately(0.09) effectively(0.09) adequately(0.09) normally(0.07) | |
| 12 | insulin | insulin(1.00) adequate(0.00) insufficient(0.00) pancreatic(0.00) oins(0.00) | |
| 13 | , | properly(0.00) effectively(0.00) correctly(0.00) adequately(0.00) appropriately(0.00) | |
| 14 | leading | leading(0.38) disrupting(0.26) thereby(0.03) impair(0.02) impacting(0.01) | |
| 15 | to | disruptions(0.00) disrupted(0.00) disturbances(0.00) esub(0.00) ürger(0.00) | |
| 16 | high | elevated(0.83) high(0.02) glucose(0.01) abnormal(0.01) abnormalities(0.00) | |
| 17 | blood | blood(0.82) levels(0.02) glucose(0.00) bloodstream(0.00) sugar(0.00) | |
| 18 | sugar | sugar(0.93) glucose(0.07) sugars(0.00) сахар(0.00) suger(0.00) | |
| 19 | levels | levels(1.00) nive(0.00) concentrations(0.00) level(0.00) elige(0.00) | |
| 20 | , | riger(0.01) ornment(0.00) gnore(0.00) elige(0.00) odyn(0.00) | |
| 21 | while | while(0.67) whereas(0.32) which(0.00) anske(0.00) aniu(0.00) | |
| 22 | insulin | insulin(1.00) proper(0.00) normal(0.00) maintaining(0.00) healthy(0.00) | |
| 23 | is | helps(0.43) regulates(0.22) itself(0.07) controls(0.02) injections(0.01) | |
| 24 | the | crucial(0.42) essential(0.29) necessary(0.02) responsible(0.02) produced(0.01) | |
| 25 | hormone | hormone(1.00) hormones(0.00) pancreatic(0.00) hormonal(0.00) horm(0.00) | |
| 26 | that | responsible(0.83) essential(0.02) crucial(0.02) produced(0.02) tasked(0.00) | |
| 27 | helps | regulates(0.92) helps(0.03) controls(0.01) regulate(0.01) lowers(0.00) | |
| 28 | regulate | regulate(0.98) regulates(0.00) regulating(0.00) cells(0.00) regulation(0.00) | |
| 29 | blood | blood(0.21) glucose(0.16) sugar(0.03) these(0.00) sugars(0.00) | |

**所有领域词**: abilities, ability, able, abnormal, abnormalities, absorption, acts, adequ, adequate, adequately, affect, affected, affecting, affects, agini, aids, allows, altering, alters, amounts
**关系词**: affects, causing, central, critical, critical, critical, critical, crucial, crucial, crucial


## calcium + bones
生成: "that calcium is a crucial mineral that helps build and maintain bone structure and density.<|im_end|"

| step | gen_token | workspace top-5 (prob) | 判断 |
|------|-----------|------------------------|------|
| 0 | that | crucial(0.27) essential(0.07) critical(0.05) vital(0.03) pivotal(0.01) | |
| 1 | calcium | calcium(0.89) adequate(0.04) sufficient(0.00) Calc(0.00) adequ(0.00) | |
| 2 | is | plays(0.02) ustum(0.02) strengthens(0.01) reon(0.01) ombine(0.01) | |
| 3 | a | essential(0.79) crucial(0.08) vital(0.01) critical(0.00) ensely(0.00) | |
| 4 | crucial | critical(0.20) crucial(0.18) essential(0.12) vital(0.11) primary(0.00) | |
| 5 | mineral | mineral(0.24) component(0.23) nutrient(0.10) structural(0.03) dietary(0.01) | |
| 6 | that | essential(0.19) ensely(0.01) crucial(0.01) integral(0.01) deposited(0.01) | |
| 7 | helps | helps(0.46) contributes(0.05) forms(0.05) builds(0.03) strengthens(0.02) | |
| 8 | build | build(0.12) maintain(0.10) strengthen(0.02) bones(0.02) compose(0.01) | |
| 9 | and | strong(0.14) bone(0.08) bones(0.05) stronger(0.03) dens(0.02) | |
| 10 | maintain | maintain(0.73) maint(0.02) maintains(0.00) sustain(0.00) maintenance(0.00) | |
| 11 | bone | bone(0.27) bones(0.05) skeletal(0.04) strong(0.02) healthy(0.02) | |
| 12 | structure | density(0.40) structure(0.20) strength(0.03) integrity(0.02) structures(0.01) | |
| 13 | and | ngth(0.00) integrity(0.00) rength(0.00) strength(0.00) oins(0.00) | |
| 14 | density | density(0.80) strength(0.01) integrity(0.00) ensity(0.00) rength(0.00) | |
| 15 | . | aniu(0.00) agini(0.00) vider(0.00) rray(0.00) ngth(0.00) | |
| 16 | <|im_end|> | aniu(0.02) owany(0.00) zych(0.00) acje(0.00) kich(0.00) | |

**所有领域词**: acje, acts, adequ, adequate, adequately, adobe, agini, aids, ample, andum, aniem, aniu, anske, anyl, apeut, architecture, asurer, atform, bahwa, bild
**关系词**: build, builds, component, component, critical, critical, critical, crucial, crucial, crucial


## blood + heart
生成: "that the heart pumps blood throughout the body, supplying it to all tissues and organs, including th"

| step | gen_token | workspace top-5 (prob) | 判断 |
|------|-----------|------------------------|------|
| 0 | that | evident(0.39) crucial(0.10) essential(0.05) vital(0.03) critical(0.02) | |
| 1 | the | blood(0.35) arteries(0.02) emean(0.01) pumping(0.00) anske(0.00) | |
| 2 | heart | heart(1.00) heartbeat(0.00) pumping(0.00) cardiovascular(0.00) blood(0.00) | |
| 3 | pumps | pumps(0.72) pump(0.01) pumping(0.00) pumped(0.00) circ(0.00) | |
| 4 | blood | blood(0.98) bloodstream(0.00) крови(0.00) oxygen(0.00) bloody(0.00) | |
| 5 | throughout | throughout(0.96) through(0.00) ensely(0.00) anske(0.00) anken(0.00) | |
| 6 | the | vascular(0.01) apesh(0.01) vessels(0.01) amac(0.01) vanished(0.01) | |
| 7 | body | body(0.87) circ(0.06) vascular(0.01) cardiovascular(0.01) bloodstream(0.00) | |
| 8 | , | riger(0.00) through(0.00) ections(0.00) circulating(0.00) emean(0.00) | |
| 9 | supplying | supplying(0.36) delivering(0.17) circulating(0.15) distributing(0.04) transporting(0.03) | |
| 10 | it | nutrients(0.40) vital(0.12) oxygen(0.10) tissues(0.09) essential(0.03) | |
| 11 | to | vital(0.03) essential(0.02) emean(0.01) ections(0.01) necessary(0.01) | |
| 12 | all | tissues(0.75) vanished(0.02) cells(0.01) organs(0.00) tissue(0.00) | |
| 13 | tissues | tissues(0.90) organs(0.05) cells(0.01) tissue(0.01) bodily(0.00) | |
| 14 | and | ections(0.05) elmet(0.04) ivors(0.01) ogui(0.01) ocre(0.00) | |
| 15 | organs | organs(0.99) organ(0.00) regulating(0.00) escort(0.00) facilitating(0.00) | |
| 16 | , | riger(0.00) emean(0.00) asurer(0.00) ocre(0.00) essential(0.00) | |
| 17 | including | includ(0.00) escort(0.00) ensuring(0.00) ections(0.00) ylko(0.00) | |
| 18 | the | vital(0.03) itself(0.03) organs(0.03) regulating(0.02) escort(0.01) | |
| 19 | adrenal | heart(0.43) adrenal(0.28) kidneys(0.11) tumor(0.05) glands(0.03) | |
| 20 | glands | glands(0.99) gland(0.01) organs(0.00) hormones(0.00) tissues(0.00) | |
| 21 | where | where(0.87) involved(0.01) responsible(0.01) which(0.01) olest(0.00) | |
| 22 | tumors | tumors(0.95) hormones(0.02) blood(0.01) hormonal(0.00) tumor(0.00) | |
| 23 | like | affecting(0.49) impacting(0.00) affect(0.00) ffect(0.00) panse(0.00) | |
| 24 | p | neuro(0.48) those(0.04) chrom(0.01) adrenal(0.01) aden(0.00) | |
| 25 | he | ancell(0.05) icester(0.05) inyin(0.04) egasus(0.03) ofil(0.03) | |
| 26 | och | chrom(0.00) ucch(0.00) umat(0.00) asant(0.00) chod(0.00) | |
| 27 | rom | amac(0.09) coma(0.06) roma(0.05) ommen(0.02) ydro(0.01) | |
| 28 | oc | amac(0.01) okit(0.00) ocab(0.00) zyst(0.00) ecast(0.00) | |
| 29 | yt | ysts(0.06) erot(0.01) serter(0.01) otron(0.00) zych(0.00) | |

**所有领域词**: abama, abnormalities, achinery, actable, acts, aden, adrenal, adrenaline, affect, affected, affecting, affects, afür, agal, agini, agus, aiding, alink, allel, allowing
**关系词**: affects, circ, circ, circ, circ, circ, circulated, circulated, circulation, circulation


## diagnosis + treatment
生成: "that diagnosis informs the specific treatment plan needed for a patient's condition.<|im_end|>"

| step | gen_token | workspace top-5 (prob) | 判断 |
|------|-----------|------------------------|------|
| 0 | that | foundational(0.08) causal(0.06) sequential(0.03) pivotal(0.02) crucial(0.01) | |
| 1 | diagnosis | diagnosis(0.99) accurate(0.00) diagnostic(0.00) diagnoses(0.00) diagnostics(0.00) | |
| 2 | informs | informs(0.55) guides(0.23) leads(0.06) determines(0.05) directs(0.00) | |
| 3 | the | targeted(0.02) personalized(0.01) tailored(0.01) reatment(0.00) precise(0.00) | |
| 4 | specific | appropriate(0.04) specifics(0.02) tailored(0.02) specificity(0.01) selection(0.01) | |
| 5 | treatment | nature(0.00) intervention(0.00) approach(0.00) course(0.00) care(0.00) | |
| 6 | plan | plan(0.61) approach(0.03) regimen(0.02) modal(0.01) plans(0.01) | |
| 7 | needed | tailored(0.69) needed(0.10) necessary(0.02) tail(0.02) tailor(0.02) | |
| 8 | for | okit(0.03) ocre(0.01) tailored(0.01) ulings(0.01) egov(0.01) | |
| 9 | a | optimal(0.36) effective(0.30) effectively(0.03) optimizing(0.03) okit(0.01) | |
| 10 | patient | individual(0.00) condition(0.00) okit(0.00) healthcare(0.00) successful(0.00) | |
| 11 | 's | reon(0.01) okit(0.01) egov(0.01) licted(0.01) agini(0.01) | |
| 12 | condition | condition(0.79) illness(0.01) healthcare(0.00) unique(0.00) conditions(0.00) | |
| 13 | . | agini(0.01) riger(0.01) ensely(0.00) omid(0.00) rror(0.00) | |
| 14 | <|im_end|> | aniu(0.00) kich(0.00) zych(0.00) ogui(0.00) alink(0.00) | |

**所有领域词**: accur, accurate, accurately, acje, addressing, adesh, agini, ailments, aimed, alink, andum, aniem, aniu, anske, appro, approach, approaches, appropri, appropriate, ardu
**关系词**: caus, causal, crucial, guided, reatment, reatment, sequential, therapeutic, therapeutic, therapy
