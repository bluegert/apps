import Head from "next/head";
import Sidebar from "../components/sidebar";
import { SetStateAction } from "react";
import MenuBarMobile from "../components/menubarmobile";
import { useAuth } from "../auth/AuthProvider";

export default function Home() {
  const { user } = useAuth();
  console.log(user);
  return (
    <div>
    <Head>
      <title>Axonn</title>
      <link rel="icon" href="/axonn.ico" />
    </Head>
    <MenuBarMobile setter={function (value: SetStateAction<boolean>): void {
        throw new Error("Function not implemented.");
      } } />
    <Sidebar show={false} setter={function (value: SetStateAction<boolean>): void {
        throw new Error("Function not implemented.");
      } } />
    </div>
  )
}
