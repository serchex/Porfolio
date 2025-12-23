'use client';

import React from 'react';
import LetterGlitch from '@/components/LetterGlitch';
import { CometCard } from "@/components/comet-card";
import { VideoText } from "@/components/video-text";
import ChromaGrid from '@/components/ChromaGrid';
import TextPressure from '@/components/TextPressure';
import BlurText from "@/components/BlurText";
import TypingText from "@/components/typing-text"
import { FloatingDockHorizontal } from "@/components/floating-dock";
import {
  IconBrandGithub,
  IconMathIntegralX,
  IconSql,
  IconBrandAws,
  IconBrandPython,
} from "@tabler/icons-react";
import TiltedCard from '@/components/TiltedCard';

const links = [
    {
      title: "SQL",
      icon: (
        <IconSql className="h-full w-full text-neutral-500 dark:text-neutral-300" />
      ),
      href: "#",
    },
 
    {
      title: "Python",
      icon: (
        <IconBrandPython className="h-full w-full text-neutral-500 dark:text-neutral-300" />
      ),
      href: "#",
    },
    {
      title: "Amazon Web Services",
      icon: (
        <IconBrandAws className="h-full w-full text-neutral-500 dark:text-neutral-300" />
      ),
      href: "#",
    },
    {
      title: "Pandas",
      icon: (
        <img
          src="https://res.cloudinary.com/dwko6puxt/image/upload/v1766435602/pandas_nxq1nj.svg"
          width={20}
          height={20}
          alt="Pandas"
        />
      ),
      href: "#",
    },
    {
      title: "TensorFlow",
      icon: (
        <img
          src="https://res.cloudinary.com/dwko6puxt/image/upload/v1766435681/tensorflow_1_hi0krq.svg"
          width={20}
          height={20}
          alt="Pandas"
        />
      ),
      href: "#",
    },
 
    {
      title: "Mathemathics",
      icon: (
        <IconMathIntegralX className="h-full w-full text-neutral-500 dark:text-neutral-300" />
      ),
      href: "#",
    },
    {
      title: "Git",
      icon: (
        <IconBrandGithub className="h-full w-full text-neutral-500 dark:text-neutral-300" />
      ),
      href: "#",
    },
    {
      title: "Linux",
      icon: (
        <img
          src="https://res.cloudinary.com/dwko6puxt/image/upload/v1766436200/linux_ep9s0c.svg"
          width={20}
          height={20}
          alt="Linux"
        />
      ),
      href: "#",
    },
    {
      title: "PowerBI",
      icon: (
        <img
          src="https://res.cloudinary.com/dwko6puxt/image/upload/v1766436484/Microsoft-Power-Bi--Streamline-Svg-Logos_mcqjfr.svg"
          width={20}
          height={20}
          alt="PowerBI"
        />
      ),
      href: "#",
    },
  ];

const handleAnimationComplete = () => {
  console.log('Animation completed!');
};

const items = [
  {
    image: "https://res.cloudinary.com/dwko6puxt/image/upload/v1766425860/github-6980894_960_720_imk1lr.webp",
    title: "Github",
    subtitle: "",
    handle: "❖",
    borderColor: "#ddddddff",
    gradient: "linear-gradient(145deg, #e0e0e0ff, #000)",
    url: "https://github.com/serchex"
  },
  {
    image: "https://res.cloudinary.com/dwko6puxt/image/upload/v1766426049/HackerRank_Icon-1000px_pi3pmy.png",
    title: "HackerRank",
    subtitle: "",
    handle: "❖",
    borderColor: "#10B981",
    gradient: "linear-gradient(180deg, #10B981, #000)",
    url: "https://www.hackerrank.com/profile/dalgagaimer"
  },
  {
    image: "https://res.cloudinary.com/dwko6puxt/image/upload/v1766426548/ChatGPT_Image_22_dic_2025_12_02_20_p.m._hne7la.png",
    title: "Linkedin",
    subtitle: "",
    handle: "❖",
    borderColor: "#006effff",
    gradient: "linear-gradient(180deg, #008afcff, #000)",
    url: "https://www.linkedin.com/in/sergio-daniel-gonzalez-lopez-1513b9226/"
  },
  {
    image: "https://res.cloudinary.com/dwko6puxt/image/upload/v1766427106/ChatGPT_Image_22_dic_2025_12_11_39_p.m._mmzn4a.png",
    title: "YouTube",
    subtitle: "",
    handle: "❖",
    borderColor: "#ff0000ff",
    gradient: "linear-gradient(180deg, #fc1d00ff, #000)",
    url: "https://www.youtube.com/@sethprograms"
  }
];

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-black">
      
      {/* Fondo */}
      <div className="absolute inset-0 ">
        <LetterGlitch
          glitchSpeed={200}
          centerVignette={true}
          outerVignette={true}
          smooth={true}
          glitchColors={["#000000ff", "#4b4b4bff", "#034603ff"]}
          characters="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        />
      </div>

      {/* Contenido encima */}
      <div className="will-change-transform transform-gpu" style={{position: 'relative', height: '250px'}}>
        <TextPressure
          text="Sergio  Gonzalez"
          flex={true}
          alpha={false}
          stroke={false}
          width={false}
          weight={true}
          italic={true}
          textColor="#ffffff"
          strokeColor="#ff0000"
          minFontSize={36}
        />
      </div>
      <div className="relative z-10 flex flex-col min-h-screen items-center justify-center text-white text-4xl font-semibold">
        <BlurText
          text="Main AI proyects"
          delay={1000}
          animateBy="words"
          direction="top"
          onAnimationComplete={handleAnimationComplete}
          className="text-2xl mb-1"
        />
        <div className='grid grid-cols-1 gap-10 sm:grid-cols-2 lg:grid-cols-3 place-items-center max-w-6xl w-full px-6'>
          <CometCard>
            <button
              type="button"
              className="my-10 flex w-80 cursor-pointer flex-col items-stretch rounded-[16px] border-0 bg-[#1F2121] p-2 saturate-0 md:my-20 md:p-4"
              aria-label="View invite F7RA"
              style={{
                transformStyle: "preserve-3d",
                transform: "none",
                opacity: 1,
              }}
            >
              <div className="mx-2 flex-1">
                <div className="relative mt-2 aspect-[4/4] w-full">
                  <img
                    loading="lazy"
                    className="absolute inset-0 h-full w-full rounded-[16px] bg-[#000000] object-cover contrast-100"
                    alt="Invite background"
                    src="https://res.cloudinary.com/dwko6puxt/image/upload/v1766372235/Image_23_qgs1hg.png"
                    style={{
                      boxShadow: "rgba(0, 0, 0, 0.05) 0px 5px 6px 0px",
                      opacity: 1,
                    }}
                  />
                </div>
              </div>
              <div className="mt-2 flex flex-shrink-0 items-center justify-between p-4 font-mono text-white">
                <div className="text-xs">CC's Fraud Detector</div>
                <div className="text-xs text-gray-300 opacity-50">CATBOST</div>
              </div>
            </button>
          </CometCard>
          <CometCard>
            <button
              type="button"
              className="my-10 flex w-80 cursor-pointer flex-col items-stretch rounded-[16px] border-0 bg-[#1F2121] p-2 saturate-0 md:my-20 md:p-4"
              aria-label="View invite F7RA"
              style={{
                transformStyle: "preserve-3d",
                transform: "none",
                opacity: 1,
              }}
            >
              <div className="mx-2 flex-1">
                <div className="relative mt-2 aspect-[4/4] w-full">
                  <img
                    loading="lazy"
                    className="absolute inset-0 h-full w-full rounded-[16px] bg-[#000000] object-cover contrast-100"
                    alt="Invite background"
                    src="https://res.cloudinary.com/dwko6puxt/image/upload/v1766372040/Image_22_dic_ld5qxb.png"
                    style={{
                      boxShadow: "rgba(0, 0, 0, 0.05) 0px 5px 6px 0px",
                      opacity: 1,
                    }}
                  />
                </div>
              </div>
              <div className="mt-2 flex flex-shrink-0 items-center justify-between p-4 font-mono text-white">
                <div className="text-xs">Emotion Detector</div>
                <div className="text-xs text-gray-300 opacity-50">BERT</div>
              </div>
            </button>
          </CometCard>
          <CometCard >
            <button
              type="button"
              className="my-10 flex w-80 cursor-pointer flex-col items-stretch rounded-[16px] border-0 bg-[#1F2121] p-2  md:my-20 md:p-4"
              aria-label="View invite F7RA"
              style={{
                transformStyle: "preserve-3d",
                transform: "none",
                opacity: 1,
              }}
            >
              <div className="mx-2 flex-1">
                <div className="relative mt-2 aspect-[4/4] w-full">
                  <img
                    loading="lazy"
                    className="absolute inset-0 h-full w-full rounded-[16px] bg-[#000000] object-cover contrast-100"
                    alt="Invite background"
                    src="https://res.cloudinary.com/dwko6puxt/image/upload/v1766371884/Image_21_dic_2025_judqly.png"
                    style={{
                      boxShadow: "rgba(0, 0, 0, 0.05) 0px 5px 6px 0px",
                      opacity: 1,
                    }}
                  />
                </div>
              </div>
              <div className="mt-2 flex flex-shrink-0 items-center justify-between p-4 font-mono text-white">
                <div className="text-xs">House Prediction Price</div>
                <div className="text-xs text-gray-300 opacity-50">CATBOOST</div>
              </div>
            </button>
          </CometCard>
        </div>
        <div className="relative h-[210px] w-full ">
            <VideoText src="https://res.cloudinary.com/dwko6puxt/video/upload/v1766377594/2025-12-21-22-09-09_ekabjz.webm">About me</VideoText>
        </div>
      </div>
      <section className="relative overflow-hidden min-h-[40vh] md:h-[5vh]">
        <div  className="relative h-full w-full">
          <ChromaGrid 
            items={items}
            radius={200}
            damping={0.45}
            fadeOut={0.6}
            ease="power3.out"
          />
        </div>
      </section>
      <section className="mt-4 flex w-full justify-center px-4">
        <div className="w-full max-w-full sm:max-w-fit">
          <div className="
  rounded-md
  bg-gray-800
  px-4
  py-3
  font-mono
  text-green-50
  shadow-lg
  text-center
  sm:text-left
">
            <TypingText
              text="> main skills"
              grow
              className="
    mx-auto
    text-base
    sm:text-lg
    md:text-2xl
    whitespace-normal
    md:whitespace-nowrap
  "
            />
          </div>
        </div>
      </section>
      <section className="mt-8 flex w-full justify-center px-4">
        <FloatingDockHorizontal
          items={links}
        />
      </section>
      <section className="w-full px-4 mt-12">
        <div
          className="
            grid
    grid-cols-1

    gap-y-4
    gap-x-6

    sm:grid-cols-2
    sm:gap-y-6

    md:grid-cols-3
    md:gap-x-50

    lg:grid-cols-4

    max-w-6xl
    mx-auto
    place-items-center
          "
        >
          {/* cards aquí */}
          <TiltedCard
            imageSrc="https://i.scdn.co/image/ab67616d0000b273d9985092cd88bffd97653b58"
            altText="Kendrick Lamar - GNX Album Cover"
            captionText="Kendrick Lamar - GNX"
            containerHeight="300px"
            containerWidth="300px"
            imageHeight="300px"
            imageWidth="300px"
            rotateAmplitude={12}
            scaleOnHover={1.2}
            showMobileWarning={false}
            showTooltip={true}
            displayOverlayContent={true}
            overlayContent={
              <p className="tilted-card-demo-text">
                Kendrick Lamar - GNX
              </p>
            }
          />
          <TiltedCard
            imageSrc="https://i.scdn.co/image/ab67616d0000b273d9985092cd88bffd97653b58"
            altText="Kendrick Lamar - GNX Album Cover"
            captionText="Kendrick Lamar - GNX"
            containerHeight="300px"
            containerWidth="300px"
            imageHeight="300px"
            imageWidth="300px"
            rotateAmplitude={12}
            scaleOnHover={1.2}
            showMobileWarning={false}
            showTooltip={true}
            displayOverlayContent={true}
            overlayContent={
              <p className="tilted-card-demo-text">
                Kendrick Lamar - GNX
              </p>
            }
          />
        </div>
      </section>
    </main>
  );
}