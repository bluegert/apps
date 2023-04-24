import {
    createContext,
    useContext,
    useEffect,
    useState,
    ReactNode,
  } from "react";
  import { supabaseClient } from "../utils/SupabaseClient";
  
  type AuthContextType = {
    user: any;
    login: (email: string, password: string) => Promise<void>;
    logout: () => void;
  };
  
  type AuthProviderProps = {
    children: ReactNode;
  };
  
  export const AuthContext = createContext<AuthContextType>({
    user: null,
    login: async () => {},
    logout: () => {},
  });
  
  export const AuthProvider = ({ children }: AuthProviderProps) => {
    // TODO: Fix user type.
    const [user, setUser] = useState<any>(null);
  
    useEffect(() => {
      retrieveSession();
    }, []);
  
    const retrieveSession = async () => {
      const { data, error } = await supabaseClient.auth.getSession();
  
      if (error) {
        console.log(error);
  
        return;
      }
  
      const session = data.session;
  
      if (session?.user) {
        setUser(session.user);
      }
    };
  
    const login = async (email: string, password: string): Promise<void> => {
      const { data, error } = await supabaseClient.auth.signInWithPassword({
        email,
        password,
      });
  
      if (error) throw new Error(error.message);
  
      setUser(data.user);
    };
  
    const logout = async () => {
      const { error } = await supabaseClient.auth.signOut();
  
      if (error) throw new Error(error.message);
  
      setUser(null);
    };
  
    return (
      <AuthContext.Provider value={{ user, login, logout }}>
        {children}
      </AuthContext.Provider>
    );
  };
  
  export const useAuth = () => useContext(AuthContext);
  