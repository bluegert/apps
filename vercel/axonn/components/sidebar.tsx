// @/components/Layout/Sidebar.js
import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'

import { SlHome } from 'react-icons/sl'
import { BsEnvelopeAt, BsPencilSquare, BsClipboardData } from 'react-icons/bs'
import { FaRegLightbulb } from 'react-icons/fa'


export default function Sidebar({ show, setter }: { show: boolean, setter: React.Dispatch<React.SetStateAction<boolean>> }) {
    const router = useRouter();

    // Define our base class
    const className = "w-[250px] transition-[margin-left] ease-in-out duration-500 fixed md:static top-0 bottom-0 left-0 z-40";
    // Append class based on state of sidebar visiblity
    const appendClass = show ? " ml-0" : " ml-[-250px] md:ml-0";

    // Clickable menu items
    const MenuItem = ({ icon, name, route }: any) => {
        // Highlight menu item based on currently displayed route
        const colorClass = router.pathname === route ? "text-white" : "text-white/50 hover:text-white";

        return (
            <Link
                href={route}
                onClick={() => {
                    setter(oldVal => !oldVal);
                }}
                className={`flex gap-1 [&>*]:my-auto text-md pl-6 py-3 border-b-[1px] border-b-white/10 ${colorClass}`}
                style={{ fontSize: "1.25rem" }} 
            >
                <div className="text-xl flex [&>*]:mx-auto w-[30px]">
                    {icon}
                </div>
                <div>{name}</div>
            </Link>
        )
    }

    // Overlay to prevent clicks in background, also serves as our close button
    const ModalOverlay = () => (
        <div
            className={`flex md:hidden fixed top-0 right-0 bottom-0 left-0 bg-black/50 z-30`}
            onClick={() => {
                setter(oldVal => !oldVal);
            }}
        />
    )

    return (
        <>
        <div className={`${className}${appendClass}`}>
                <div className="p-2 flex">
                    <Link href="/">
                    <div className="flex gap-2 items-center p-2">
    <img src="/icon_white.png" alt="Company Logo" width={50} height={50} />
    <span style={{ fontSize: "2rem", fontWeight: "bold" }}>Axonn</span>
  </div>
                    </Link>
                </div>
                <div className="flex flex-col">
                    <MenuItem
                        name="Home"
                        route="/"
                        icon={<SlHome />}
                    />
                    <MenuItem
                        name="Ideas"
                        route="/ideas"
                        icon={<FaRegLightbulb />}
                    />
                    <MenuItem
                        name="Create"
                        route="/create"
                        icon={<BsPencilSquare />}
                    />
                    <MenuItem
                        name="Strategy"
                        route="/strategy"
                        icon={<BsClipboardData />}
                    />
                    <MenuItem
                        name="Contact"
                        route="/contact"
                        icon={<BsEnvelopeAt />}
                    />
                </div>
            </div>
            {show ? <ModalOverlay /> : <></>}
        </>
    )
}
