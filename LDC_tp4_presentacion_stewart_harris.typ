#import "@preview/touying:0.5.5": *
#import "@preview/clean-math-presentation:0.1.1": *
#import "@preview/subpar:0.2.2"

#set image(height: 40%)
#set grid(gutter: 10pt)

#show: clean-math-presentation-theme.with(
  config-info(
    title: [Laboratorio de comunicaciones\
    TP4: Simulación sistema óptico MI/DD],
    short-title: [Simulación sistema óptico MI/DD],
    authors: (
      (name: "María Luz Stewart Harris")),
    author: "María Luz Stewart Harris",
    date: datetime(year: 2025, month: 10, day: 23),
  ),
  config-common(
    slide-level: 2,
    //handout: true,
    //show-notes-on-second-screen: right,
  ),
  progress-bar: true,
  align: horizon
)

#title-slide(
  logo1: image("figs/Instituto-Balseiro.png", height: 4.5em),
)


= Descripción sistema


#focus-slide[
  Sistema modelado
]

#slide(title: "Descripción sistema modelado")[
#figure(
  image("figs/componentes.png"),
  caption: [Sistema modelado]
)
]

= Impacto de la amplitud de modulación

#focus-slide[
  Impacto de la amplitud de modulación
]


#slide()[
  #grid(align: (center+horizon, center+horizon), columns: (1fr, 1fr),
  subpar.grid(
    columns: (1fr, 1fr),
  figure(image("figs/ej1_pd_eye_VppOverVpi_0.5.png"), caption: [$V_(p p) = 0.5 V_pi$]), <mzm_0.5>,
  figure(image("figs/ej1_pd_eye_VppOverVpi_0.8.png"), caption: [$V_(p p) = 0.8 V_pi$]), <mzm_0.8>,
  figure(image("figs/ej1_pd_eye_VppOverVpi_1.0.png"), caption: [$V_(p p) = 1.0 V_pi$]), <mzm_1.0>,
  figure(image("figs/ej1_pd_eye_VppOverVpi_1.2.png"), caption: [$V_(p p) = 1.2 V_pi$]), <mzm_1.2>,
  ),
  table(
    columns: (auto, auto),
    inset: 10pt,
    align: horizon+center,
    table.header(
      [*$V_(p p)$ de modulación*], [*Apertura diagrama de ojo*],
    ),
    $0.5 V_(pi)$, $2.1 m V$,
    $0.8 V_(pi)$, $3.1 m V$,
    $1.0 V_(pi)$, $3.5 m V$,
    $1.2 V_(pi)$, $3.1 m V$,
  )
  )
]

#slide()[
  #figure(
    image("figs/ej1_mzm_transfer_function.svg", width: 60%),
    caption: [Transferencia de potencia del MZM en función de la modulación $u(t)$ ($V_"bias" = -2.5V$).]
  )<mzm_trans>
]


= Impacto de la longitud de la fibra óptica

#focus-slide[
  Impacto de la longitud de la fibra óptica
]

#slide(title:"Impacto de la longitud de la fibra óptica")[
  #grid(align: (center+horizon, center+horizon), columns: (1fr, 1fr),
  subpar.grid(
    columns: (1fr, 1fr),
    figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_40.png"), caption: [$L=40 k m$]), <L_40>,
    figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_60.png"), caption: [$L=60 k m$]), <L_60>,
    figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_80.png"), caption: [$L=80 k m$]), <L_80>,
    figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_100.png"), caption: [$L=100 k m$]), <L_100>,
  ),
  $ 
    P_"láser" &= -10 "dBm" \
    alpha&=0 "dB/km" \
    beta_2&=-20 (p s^2)/(k m) \
    beta_3&=0.1 (p s^3)/(k m) \
    gamma&=0 
  $
  )
]

#slide(title:"Impacto de la longitud de la fibra óptica")[
  #grid(
    align: (center+horizon, center+horizon), columns: (1fr, 1fr),
    subpar.grid(
      columns: (1fr, 1fr),
      figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_40.png"), caption: [$L=40 k m$]), <L_40>,
      figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_60.png"), caption: [$L=60 k m$]), <L_60>,
      figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_80.png"), caption: [$L=80 k m$]), <L_80>,
      figure(image("figs/ej2_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_0_L_100.png"), caption: [$L=100 k m$]), <L_100>,
      
      label: <L>,
    ),
    table(
      columns: (auto, auto),
      inset: 10pt,
      align: horizon+center,
      table.header(
        [*Longitud fibra óptica*], [*Apertura diagrama de ojo*],
      ),
      $40 k m$, $2.9 m V$,
      $60 k m$, $2.2 m V$,
      $80 k m$, $2.0 m V$,
      $100 k m$, $0.7 m V$,
    )
  )
]

#slide(title:"Impacto de la longitud de la fibra óptica")[
  $
    L_(d 2) &= T_0^2/abs(beta_2) &= (100 p s)^2/(20 (p s^2) / (k m)) &= 500 k m \
    L_(d 3) &= T_0^3/abs(beta_3) &= (100 p s)^3/(0.1 (p s^3) / (k m)) &= 10^7 k m >> L_(d 2)\
  $
]


= Impacto de la potencia del láser

#focus-slide[
  Impacto de la potencia del láser
]

#slide()[
  #grid(align: (center+horizon, center+horizon), columns: (1fr, 1fr),
  subpar.grid(
    columns: (1fr, 1fr),
    figure(image("figs/ej3_pd_eye_P_-10_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=-10 "dBm"$]), <P_-10>,
    figure(image("figs/ej3_pd_eye_P_0_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=0"dBm"$]), <P_0>,
    figure(image("figs/ej3_pd_eye_P_10_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=10 "dBm"$]), <P_10>,
    figure(image("figs/ej3_pd_eye_P_20_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=20 "dBm"$]), <P_20>,
  ),
  $ 
    L &= 40 "km" \
    alpha&=0 "dB/km" \
    beta_2&=-20 (p s^2)/(k m) \
    beta_3&=0 (p s^3)/(k m) \
    gamma&=1.5
  $
  )
]

#slide()[
  #grid(align: (center+horizon, center+horizon), columns: (1fr, 1fr),
  subpar.grid(
    columns: (1fr, 1fr),
    figure(image("figs/ej3_pd_eye_P_-10_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=-10 "dBm"$]), <P_-10>,
    figure(image("figs/ej3_pd_eye_P_0_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=0"dBm"$]), <P_0>,
    figure(image("figs/ej3_pd_eye_P_10_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=10 "dBm"$]), <P_10>,
    figure(image("figs/ej3_pd_eye_P_20_a_0_b2_-20_b3_0_gamma_1.5_L_40.png"), caption: [$P=20 "dBm"$]), <P_20>,
  ),
  table(
    columns: (auto, auto),
    inset: 10pt,
    align: horizon+center,
    table.header(
      [*$P_"láser"$*], [*Apertura diagrama de ojo*],
    ),
    $-10 "dBm"$, $3.2 m V$,
    $"   "0 "dBm"$, $34 m V$,
    $" "10 "dBm"$, $350 m V$,
    $" "20 "dBm"$, $1800 m V$,
  )
  )
]

#slide()[
  #align(center)[
  #table(
    columns: (auto, auto),
    inset: 10pt,
    align: horizon+center,
    table.header(
      [*$P_"láser"$*], [*Apertura diagrama de ojo*],
    ),
    $-10 "dBm"$, $3.2 m V$,
    $"   "0 "dBm"$, $34 m V$,
    $" "10 "dBm"$, $350 m V$,
    $" "20 "dBm"$, $1800 m V$,
  )
  ]
]

= Impacto tasa de transmisión

#focus-slide[
  Impacto tasa de transmisión
]


#slide()[
  #grid(align: (center+horizon, center+horizon), columns: (1fr, 1fr),
  subpar.grid(
    columns: (1fr, 1fr),
    figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_5000000000.0.png"), caption: [$B_r=5 G b p s$]), <Br_5G>,
    figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_10000000000.0.png"), caption: [$B_r=10 G b p s$]), <Br_10G>,
    figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_15000000000.0.png"), caption: [$B_r=15 G b p s$]), <Br_15G>,
    figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_20000000000.0.png"), caption: [$B_r=20 G b p s$]), <Br_20>,
  ),
  $ 
    L &= 50 "km" \
    alpha&=0 "dB/km" \
    beta_2&=-20 (p s^2)/(k m) \
    beta_3&=0.1 (p s^3)/(k m) \
    gamma&=0
  $
  )
]

#slide()[
  #grid(align: (center+horizon, center+horizon), columns: (1fr, 1fr),
  subpar.grid(
    columns: (1fr, 1fr),
    figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_5000000000.0.png"), caption: [$B_r=5 G b p s$]), <Br_5G>,
    figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_10000000000.0.png"), caption: [$B_r=10 G b p s$]), <Br_10G>,
    figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_15000000000.0.png"), caption: [$B_r=15 G b p s$]), <Br_15G>,
    figure(image("figs/ej4_pd_eye_P_10_a_0_b2_-20_b3_0.1_gamma_0_L_50_Br_20000000000.0.png"), caption: [$B_r=20 G b p s$]), <Br_20>,
  ),
  [
    #table(
      columns: (auto, auto),
      inset: 10pt,
      align: horizon+center,
      table.header(
        [*Tasa de transmisión*], [*Apertura diagrama de ojo*],
      ),
      $5 G b p s$, $300 m V$,
      $10 G b p s$, $300 m V$,
      $15 G b p s$, $90 m V$,
      $20 G b p s$, $20 m V$,
    )\
    Nota: la apertura horizontal de diagrama de ojo de 5 Gbps es mayor que la de 10 Gbps. 
  ]
  )
]

#slide()[
  #align(center)[
    #table(
      columns: (auto, auto, auto),
      inset: 10pt,
      align: horizon+center,
      table.header(
        [*Tasa de transmisión*], [*Tiempo de pulso*], [*$L_(d 2)$*],
      ),
      $5 G b p s$, $200 p s$, $2000 k m$,
      $10 G b p s$, $100 p s$, $500 k m$,
      $15 G b p s$, $66.6 p s$, $222 k m$,
      $20 G b p s$, $50 p s$, $125 k m$,
    )
  ]
]

= Comparación modulación NRZ y RZ

#focus-slide[
  Comparación modulación NRZ y RZ
]


#slide()[
  #subpar.grid(
    columns: (1fr, 1fr, 1fr),
    figure(image("figs/ej5_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_1.5_L_40_mod_RZ.png"), caption: [\
    Modulación RZ\
    $P_"láser"=-10 d B m $\
    $P_"media"=-16 d B m $]), <mod_RZ_-10>,
    figure(image("figs/ej5_pd_eye_P_-10_a_0_b2_-20_b3_0.1_gamma_1.5_L_40_mod_NRZ.png"), caption: [\
    Modulación NRZ\
    $P_"láser"=-10 d B m $\
    $P_"media"=-13 d B m $]), <mod_NRZ_-10>,
    figure(image("figs/ej5_pd_eye_P_-7_a_0_b2_-20_b3_0.1_gamma_1.5_L_40_mod_RZ.png"), caption: [\
    Modulación RZ\
    $P_"láser"=-7 d B m $\
    $P_"media"=-13 d B m $]), <mod_RZ_-7>,
  )

  $ 
    L &= 50 "km", 
    alpha&=0 "dB/km", 
    beta_2&=-20 (p s^2)/(k m), 
    beta_3&=0.1 (p s^3)/(k m), 
    gamma&=0
  $

]

#slide()[
  #align(center)[
    #table(
      columns: (auto, auto, auto, auto, auto),
      inset: 10pt,
      align: horizon+center,
      table.header(
        [*Modulación*],[*$P_"láser"$*],[*$P_"media"$*],[*Apertura vertical\
        diagrama de ojo*],
        [*Apertura horizontal\
        diagrama de ojo*],
      ),
      [RZ],$-10 "dBm"$, $-16"dBm"$, [1.2mV], [0.78],
      [NRZ],$-10 "dBm"$, $-13"dBm"$, [3.0mV], [0.69],
      [RZ],$-7 "dBm"$, $-13"dBm"$, [3.0mV], [0.85],
    )
  ]
]

#focus-slide[
  ¡Muchas gracias!
]

#show: appendix

= Appendix

#focus-slide("Apéndice")

#slide(title: "Parámetros default utilizados")[
#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Parámetro láser*], [*Valor*],
  ),
  $P_"out"$, "-10dBm",
  $delta nu$, [10 MHz],
  [RIN], [-150dB/Hz],
),

#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Parámetro MZM*], [*Valor*],
  ),
  $V_pi$, [5V],
  $V_"bias"$, [-2.5V],
  [K], [0.8],
  [Er], [30 dB],
),

#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Parámetro PD*], [*Valor*],
  ),
  $R$, [1A/W],
  $B$, [5 GHz],
  $T$, [300 K],
  $R_l$, $50 Omega$,
  $i_d$, [10nA],
  $F_n$, [0 dB],
),

#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon+center,
  table.header(
    [*Parámetro señal*], [*Valor*],
  ),
  [Modulación], [NRZ],
  $B_r$, [5 Gbps],
)

]
